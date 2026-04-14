import os
import warnings
import cv2
import numpy as np
import sknw
from skimage.morphology import skeletonize
import networkx as nx
from shapely.geometry import LineString
from scipy.ndimage import distance_transform_edt
import apls  # 确保 apls.py 在同目录下

# ==========================================
# 0. 可选：屏蔽 pyproj 的无关警告
#    你现在是纯 PNG 像素评估，不做投影转换，这个警告可忽略
# ==========================================
warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.")

# ==========================================
# 1. 文件夹批量处理配置（请修改这里）
#    注意：必须是“掩膜图”文件夹，不是原始遥感图文件夹
# ==========================================
GT_DIR = r'E:\apls-master\data\masks'        # 真值掩膜目录
PRED_DIR = r'E:\apls-master\data\images' # 预测掩膜目录（请改成你的预测结果目录）

BIN_THRESHOLD = 127 # 二值化阈值
TOLERANCE_PIXELS = 5 # 匹配容差（像素）
MAX_SNAP_DIST = 5 # APLS 吸附距离（像素）
MIN_PATH_LENGTH = 15  # 最短路径长度阈值（像素）

# 2. 基础工具函数
# ==========================================
def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

def load_binary_mask(image_path, threshold=127):
    """
    读取灰度图并转成二值掩膜：
    前景=1, 背景=0
    """
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"图片读取失败: {image_path}")

    _, binary = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    return binary

def extract_graph_and_mask(image_path, threshold=127):
    """
    从二值掩膜中提取：
    - binary: 二值图（0/1）
    - ske: 骨架图（0/1）
    - graph: sknw 图
    """
    binary = load_binary_mask(image_path, threshold=threshold)

    # 如果整张图全黑，返回空图
    if np.sum(binary) == 0:
        return nx.MultiGraph(), binary, np.zeros_like(binary, dtype=np.uint8)

    ske = skeletonize(binary > 0).astype(np.uint16)

    # 骨架可能依然为空
    if np.sum(ske) == 0:
        return nx.MultiGraph(), binary, ske

    graph = sknw.build_sknw(ske)
    return graph, binary, ske

def adapt_graph_for_apls(graph):
    """
    将 sknw 图转成 apls 可接受的 MultiGraph 格式。
    坐标单位：像素，不是地理坐标。
    """
    G = nx.MultiGraph()

    # 空图直接返回
    if graph is None or len(graph.nodes()) == 0:
        return G

    for node, data in graph.nodes(data=True):
        y, x = data['o']
        G.add_node(node, x=float(x), y=float(y))

    for u, v, data in graph.edges(data=True):
        pts = data.get('pts', None)

        if pts is not None and len(pts) > 1:
            line = LineString([(float(pt[1]), float(pt[0])) for pt in pts])
            length = float(line.length)
        else:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            line = LineString([(x1, y1), (x2, y2)])
            length = float(line.length)

        # apls 需要 length；speed/travel_time 是兼容性字段
        G.add_edge(
            u, v,
            key=0,
            geometry=line,
            length=length,
            inferred_speed_mps=15.0,
            travel_time_s=length / 15.0 if length > 0 else 0.0
        )

    return G

def compute_completeness(bin_gt, bin_pred, tolerance=5):
    """
    Completeness:
    GT骨架上有多少比例能在预测图中找到距离 <= tolerance 的匹配
    单位：像素
    """
    ske_gt = skeletonize(bin_gt > 0)

    total_gt = np.sum(ske_gt)
    if total_gt == 0:
        # 如果真值本身没路，可视作跳过或记 1.0
        return 1.0

    dist_pred = distance_transform_edt(bin_pred == 0)
    matched_gt = np.sum((dist_pred <= tolerance) & ske_gt)

    return matched_gt / total_gt

def compute_connectivity_metric(G_gt, G_pred):
    """
    简化版 Connectivity 指标：
    比较预测图和GT图的连通分量数量
    """
    if len(G_gt.nodes()) == 0 and len(G_pred.nodes()) == 0:
        return 1.0
    if len(G_gt.nodes()) > 0 and len(G_pred.nodes()) == 0:
        return 0.0
    if len(G_gt.nodes()) == 0 and len(G_pred.nodes()) > 0:
        return 0.0

    conn_gt = nx.number_connected_components(G_gt)
    conn_pred = nx.number_connected_components(G_pred)

    if conn_pred == 0:
        return 0.0

    return min(1.0, conn_gt / conn_pred)

def compute_apls_score(G_gt_apls, G_pred_apls, max_snap_dist=5, min_path_length=15):
    """
    计算像素空间下的 APLS
    若图太小、空图或路径不足，则抛异常由外层处理
    """
    if len(G_gt_apls.nodes()) < 2 or len(G_pred_apls.nodes()) < 2:
        raise ValueError("图节点过少，无法计算 APLS")

    results = apls.make_graphs(
        G_gt_apls,
        G_pred_apls,
        weight='length',
        speed_key='inferred_speed_mps',
        travel_time_key='travel_time_s',
        max_snap_dist=max_snap_dist,
        verbose=False
    )

    (G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime,
     control_points_gt, control_points_prop,
     all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
     all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime) = results

    if len(control_points_gt) == 0 or len(control_points_prop) == 0:
        raise ValueError("控制点为空，无法计算 APLS")

    score_1, _, _, _ = apls.path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=[cp[0] for cp in control_points_gt],
        min_path_length=min_path_length,
        normalize=True
    )

    score_2, _, _, _ = apls.path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=[cp[0] for cp in control_points_prop],
        min_path_length=min_path_length,
        normalize=True
    )

    return 0.5 * (score_1 + score_2)

# ==========================================
# 3. 路径检查
# ==========================================
if not os.path.isdir(GT_DIR):
    raise FileNotFoundError(f"GT_DIR 不存在: {GT_DIR}")

if not os.path.isdir(PRED_DIR):
    raise FileNotFoundError(f"PRED_DIR 不存在: {PRED_DIR}")

img_names = sorted([f for f in os.listdir(GT_DIR) if is_image_file(f)])

if len(img_names) == 0:
    raise RuntimeError(f"在 GT_DIR 中未找到图像文件: {GT_DIR}")

# ==========================================
# 4. 初始化统计量
# ==========================================
total_completeness = 0.0
total_connectivity = 0.0
total_apls = 0.0

valid_images = 0
apls_valid_images = 0

print(f"📁 找到 {len(img_names)} 张测试图像，开始批量评估...\n")

# ==========================================
# 5. 遍历计算
# ==========================================
for img_name in img_names:
    gt_path = os.path.join(GT_DIR, img_name)
    pred_path = os.path.join(PRED_DIR, img_name)

    if not os.path.exists(pred_path):
        print(f"⚠️ 缺少对应预测图，跳过: {img_name}")
        continue

    print(f"🔄 正在处理图像: {img_name} ... ", end="")

    try:
        # A. 提取图和掩膜
        graph_gt, bin_gt, ske_gt = extract_graph_and_mask(gt_path, BIN_THRESHOLD)
        graph_pred, bin_pred, ske_pred = extract_graph_and_mask(pred_path, BIN_THRESHOLD)

        # B. Completeness
        completeness_val = compute_completeness(
            bin_gt, bin_pred, tolerance=TOLERANCE_PIXELS
        )
        total_completeness += completeness_val

        # C. 转换为 apls 图
        G_gt_apls = adapt_graph_for_apls(graph_gt)
        G_pred_apls = adapt_graph_for_apls(graph_pred)

        # D. Connectivity
        conn_metric = compute_connectivity_metric(G_gt_apls, G_pred_apls)
        total_connectivity += conn_metric
        valid_images += 1

        # E. APLS
        try:
            apls_score = compute_apls_score(
                G_gt_apls,
                G_pred_apls,
                max_snap_dist=MAX_SNAP_DIST,
                min_path_length=MIN_PATH_LENGTH
            )
            total_apls += apls_score
            apls_valid_images += 1

            print(
                f"[完成] "
                f"C: {completeness_val:.3f}, "
                f"Conn: {conn_metric:.3f}, "
                f"APLS: {apls_score:.3f}"
            )

        except Exception as apls_e:
            print(
                f"[APLS跳过] "
                f"C: {completeness_val:.3f}, "
                f"Conn: {conn_metric:.3f} "
                f"(原因: {str(apls_e)})"
            )

    except Exception as e:
        print(f"❌ 处理失败，跳过。原因: {str(e)}")

# ==========================================
# 6. 汇总结果
# ==========================================
print("\n" + "=" * 60)

if valid_images > 0:
    print(f"   有效图像数: {valid_images}")
    print(f"   ➤ 平均 Completeness : {total_completeness / valid_images:.4f}")
    print(f"   ➤ 平均 Connectivity : {total_connectivity / valid_images:.4f}")
else:
    print("   没有成功完成基础评估的图像。")

if apls_valid_images > 0:
    print(f"   ➤ 平均 APLS         : {total_apls / apls_valid_images:.4f} (基于 {apls_valid_images} 张有效图)")
else:
    print("   ➤ 平均 APLS         : 无有效结果（可能图过小、全空或拓扑不足）")

print("=" * 60)