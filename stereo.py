import cv2
import numpy as np
import open3d as o3d

DY_CORRECT   = -7.2    # вертикальний зсув між знімками (пікселі, масштаб 900px)
BASELINE_PX  = 112.0   # горизонтальний стерео базис (пікселі, масштаб 900px)
FOCAL_FULL   = 1746.0  # фокусна відстань при повній роздільності 3024px
TARGET_W     = 900     # робоча ширина

LEFT_IMG  = "./photo_2026-05-26_20-29-18.jpg"
RIGHT_IMG = "./photo_2026-05-26_20-29-20.jpg"
OUTPUT_POINTCLOUD = "./point_cloud_v4.ply"

img_L_raw = cv2.imread(LEFT_IMG)
img_R_raw = cv2.imread(RIGHT_IMG)

scale = TARGET_W / img_L_raw.shape[1]
W = int(img_L_raw.shape[1] * scale)
H = int(img_L_raw.shape[0] * scale)
focal = FOCAL_FULL * scale

img_L = cv2.resize(img_L_raw, (W, H), interpolation=cv2.INTER_AREA)
img_R = cv2.resize(img_R_raw, (W, H), interpolation=cv2.INTER_AREA)
print(f"Розмір: {W}x{H} | focal={focal:.0f}px | baseline={BASELINE_PX:.0f}px")

# Додає вирівнювання
M = np.float32([[1, 0, 0], [0, 1, DY_CORRECT]])
img_R_aligned = cv2.warpAffine(img_R, M, (W, H))

gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R_aligned, cv2.COLOR_BGR2GRAY)

# Disparity map
num_disp = 160
block    = 11

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block,
    P1=8*3*block**2,
    P2=32*3*block**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=200,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disp = stereo.compute(gray_L, gray_R).astype(np.float32) / 16.0
disp = np.nan_to_num(disp)
disp = cv2.bilateralFilter(
    np.clip(disp, 0, 255).astype(np.uint8), 9, 75, 75
).astype(np.float32)

valid = disp > 2.0
disp_color = cv2.applyColorMap(
    np.clip((disp / disp.max()) * 255, 0, 255).astype(np.uint8),
    cv2.COLORMAP_TURBO
)

# Створюємо Point Cloud
cx, cy = W / 2.0, H / 2.0
left_margin = int(BASELINE_PX * 1.1)

roi = np.zeros((H, W), dtype=bool)
roi[10:H-10, left_margin:W-10] = True
mask = valid & roi & (disp < num_disp - 5)

rows, cols = np.mgrid[0:H, 0:W]
Z = np.where(mask, (focal * BASELINE_PX) / (disp + 1e-6), 0)
X = np.where(mask, (cols - cx) * Z / focal, 0)
Y = np.where(mask, (rows - cy) * Z / focal, 0)

points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
colors = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)[mask].astype(np.float64) / 255.0

# Фільтрація викидів
z = points[:, 2]
keep = (z > np.percentile(z, 1)) & (z < np.percentile(z, 99))
points, colors = points[keep], colors[keep]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
pcd = pcd.voxel_down_sample(voxel_size=2.0)

# Правильна орієнтація (горщик внизу, листя вгорі)
pts = np.asarray(pcd.points).copy()
cls = np.asarray(pcd.colors).copy()
pts_final = np.column_stack([pts[:,0], pts[:,1], pts[:,2]])

pcd_final = o3d.geometry.PointCloud()
pcd_final.points = o3d.utility.Vector3dVector(pts_final)
pcd_final.colors = o3d.utility.Vector3dVector(cls)
o3d.io.write_point_cloud(OUTPUT_POINTCLOUD, pcd_final)
print(f"Point cloud: {len(pcd_final.points):,} точок → {OUTPUT_POINTCLOUD}")

# Візуалізація за допомоги OPEN3D
vis = o3d.visualization.Visualizer()
vis.create_window("Монстера 3D", width=1000, height=800)
vis.add_geometry(pcd_final)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=300))

opt = vis.get_render_option()
opt.background_color = np.array([0.1, 0.1, 0.15])
opt.point_size = 2.0

ctr = vis.get_view_control()
ctr.set_front([0.05, 0.1, -1])
ctr.set_up([0, -1, 0])
ctr.set_lookat([0, 0, 700])
ctr.set_zoom(0.001)

vis.run()
vis.destroy_window()