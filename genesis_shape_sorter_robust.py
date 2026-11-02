"""Genesis World: silhouette-based waste sorting, Franka arm, Colab prototype.

- RGB color is deliberately identical across three objects.
- Segmentation IDs identify object pixels (privileged simulator information), NOT its class.
- RandomForest learns silhouette descriptors from camera views of objects.
- Uses an ideal suction/weld constraint, not frictional finger grasp.
- Training and evaluation views involve the same three synthetic objects; NOT generalization to real waste.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import genesis as gs

LABELS = ['plastic_bottle', 'paper', 'metal_can']
Y = [-0.20, 0.0, 0.20]
DROP_Y = [-0.40, 0.0, 0.40]
PICKUP_X, DROP_X = 0.62, 0.32
NEUTRAL = '0.63 0.65 0.67 1'
OUT = Path('genesis_robust_outputs')

# Class index -> destination bin (same order as LABELS).
BIN_COLORS = [(0.12, 0.42, 0.92), (0.15, 0.72, 0.28), (0.98, 0.49, 0.10)]
BIN_NAMES = ['PLASTIC / BLUE', 'PAPER / GREEN', 'METAL / ORANGE']
BIN_W, BIN_D, BIN_H = 0.18, 0.18, 0.105
WALL = 0.012
BASE_H = 0.012



def make_urdf(kind: str, path: Path):
    """Generate simple one-link objects; geometry determines category, NOT material color."""
    if kind == 'plastic_bottle':
        # Bottle body, shoulder, and narrower neck. Baseline remains at ground contact.
        shapes = [
            ('cylinder', (0.0, 0.0, 0.033), (0.030, 0.066)),
            ('cylinder', (0.0, 0.0, 0.075), (0.022, 0.018)),
            ('cylinder', (0.0, 0.0, 0.093), (0.013, 0.018)),
        ]
    elif kind == 'metal_can':
        shapes = [('cylinder', (0., 0., 0.046), (0.033, 0.092))]
    else:
        shapes = [('box', (0., 0., 0.013), (0.080, 0.058, 0.026))]
    parts = []
    for shape, pos, dims in shapes:
        if shape == 'cylinder':
            tag = f'<cylinder radius="{dims[0]}" length="{dims[1]}"/>'
        else:
            tag = '<box size="' + ' '.join(map(str, dims)) + '"/>'
        org = ' '.join(map(str, pos))
        parts.append(f'<visual><origin xyz="{org}"/><geometry>{tag}</geometry><material name="neutral"><color rgba="{NEUTRAL}"/></material></visual>')
        parts.append(f'<collision><origin xyz="{org}"/><geometry>{tag}</geometry></collision>')
    xml = f'''<?xml version="1.0"?>
<robot name="{kind}"><link name="object"><inertial><origin xyz="0 0 0.045"/>
<mass value="0.04"/><inertia ixx="0.00004" ixy="0" ixz="0" iyy="0.00004" iyz="0" izz="0.00004"/></inertial>
{''.join(parts)}</link></robot>'''
    path.write_text(xml, encoding='utf-8')


def features(binary):
    """Contour/Hu moments and vertical width profile, invariant to uniform color."""
    mask = (np.asarray(binary) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 5 or h < 5:
        return None
    obj = np.zeros_like(mask)
    cv2.drawContours(obj, [c], -1, 255, -1)
    roi = obj[y:y+h, x:x+w]
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    profile = []
    for k in range(6):
        segment = roi[int(k*h/6):max(int((k+1)*h/6), int(k*h/6)+1)]
        widths = np.count_nonzero(segment, axis=1)
        profile.append(float(np.mean(widths) / max(w, 1)))
    perimeter = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    return np.array([h/max(w,1), area/max(w*h,1), 4*np.pi*area/max(perimeter**2, 1),
                     *np.clip(hu, -12, 12), *profile], dtype=float)


def setup_scene(quick=False):
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02),
        rigid_options=gs.options.RigidOptions(max_dynamic_constraints=8),
        vis_options=gs.options.VisOptions(segmentation_level='link'),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'), vis_mode='collision')
    paths = []
    entities = []
    for i, label in enumerate(LABELS):
        fp = OUT / f'{label}.urdf'
        make_urdf(label, fp)
        paths.append(fp)
        item = scene.add_entity(gs.morphs.URDF(file=str(fp.resolve()),
                                               pos=(PICKUP_X, Y[i], 0.001),
                                               prioritize_urdf_material=True))
        entities.append(item)
    # Create actual open-top bins: bottom and four separate rigid fixed walls.
    # Different bin colours encode the destination; all waste items stay grey.
    for i, yc in enumerate(DROP_Y):
        color = BIN_COLORS[i]
        def box(size, pos, tint):
            return scene.add_entity(
                gs.morphs.Box(size=size, pos=pos, fixed=True),
                surface=gs.surfaces.Plastic(color=tint),
            )
        box((BIN_W, BIN_D, BASE_H), (DROP_X, yc, BASE_H/2), color)
        # Side walls leave an unobstructed mouth with internal space for each object.
        for xsign in (-1, 1):
            box((WALL, BIN_D, BIN_H),
                (DROP_X + xsign*(BIN_W/2-WALL/2), yc, BASE_H + BIN_H/2), color)
        for ysign in (-1, 1):
            box((BIN_W-2*WALL, WALL, BIN_H),
                (DROP_X, yc + ysign*(BIN_D/2-WALL/2), BASE_H + BIN_H/2), color)
        # A small vertical matching marker behind the bin makes its position
        # easy to spot from the movie camera; textual labels are added to video.
        box((0.13, 0.012, 0.075),
            (DROP_X-0.035, yc + BIN_D/2 + 0.024, 0.055), color)
    vision = scene.add_camera(res=(480, 360), pos=(1.2, -0.15, 0.48),
                              lookat=(PICKUP_X, 0, 0.045), fov=37, GUI=False)
    movie = None if quick else scene.add_camera(res=(640, 480),
                      pos=(1.7, -1.4, 1.65), lookat=(0.4,0.,0.16), fov=48, GUI=False)
    scene.build()
    robot.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
    robot.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]))
    robot.set_dofs_force_range(np.array([-87,-87,-87,-87,-12,-12,-12,-100,-100]),
                               np.array([87,87,87,87,12,12,12,100,100]))
    robot.set_qpos((-1.0124,1.5559,1.3662,-1.6878,-1.5799,1.7757,1.4602,0.04,0.04))
    for _ in range(12):
        scene.step()
    return scene, robot, entities, vision, movie


def photograph(vision, entity, y, off, savepath=None):
    # View from side; slightly varied position for training and withheld test pose.
    vision.set_pose(pos=(1.18, y - 0.42 + off, 0.33 + abs(off)*0.25),
                    lookat=(PICKUP_X, y, 0.05))
    rgb, _, seg, _ = vision.render(rgb=True, segmentation=True)
    idx = int(entity.get_link('object').idx)
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[:,:,0]
    mask = (seg == idx).astype(np.uint8)
    # Segmentation only locates the object; the label isn't passed to the classifier.
    feat = features(mask)
    if feat is None:
        values, counts = np.unique(seg, return_counts=True)
        raise RuntimeError(f'Empty object mask for link {idx}. Segmentation ids: '
                           f'{dict(zip(values.tolist(), counts.tolist()))}')
    if savepath:
        overlay = cv2.cvtColor(np.asarray(rgb)[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (10, 250, 10), 2)
        cv2.imwrite(str(savepath), overlay)
    return feat



def make_labeled_video(source: Path, target: Path):
    """Overlay a fixed clear legend on the captured MP4 (requires OpenCV ffmpeg)."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print('Could not open video for labeling:', source)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (width, height))
    if not writer.isOpened():
        print('Video writer unavailable; original MP4 remains:', source)
        cap.release()
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # A dark card and color-coded labels (text is not part of classifier).
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (min(width-8, 335), 104), (24, 24, 24), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, 'AI SHAPE-BASED WASTE SORTING', (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        for i, label in enumerate(BIN_NAMES):
            r,g,b = BIN_COLORS[i]
            bgr = (int(b*255), int(g*255), int(r*255))
            y = 45 + 22*i
            cv2.rectangle(frame, (18, y-9), (31, y+3), bgr, -1)
            cv2.putText(frame, label, (39, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (255,255,255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    print('Labeled video:', target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='No video; fast arm movement')
    ap.add_argument('--speed', type=float, default=2.9, help='Motion duration multiplier; default aims for >60s')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scene, robot, items, vision, movie = setup_scene(args.quick)
    # 12 train viewpoints per object; infer from a held-out viewpoint. Labels are supplied
    # for training only, not from the segmentation ID at inference time.
    X, target = [], []
    for i, item in enumerate(items):
        for offset in np.linspace(-0.09, 0.09, 12):
            X.append(photograph(vision, item, Y[i], float(offset)))
            target.append(i)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(np.asarray(X), target)
    preds = []
    for i, item in enumerate(items):
        feature = photograph(vision, item, Y[i], 0.125,
                             savepath=OUT / f'shape_{LABELS[i]}.png')
        predicted = int(clf.predict([feature])[0])
        confidence = float(np.max(clf.predict_proba([feature])))
        preds.append((predicted, confidence))
        print(f'[Vision] {LABELS[i]} -> {LABELS[predicted]} (confidence={confidence:.2f})')
    score = accuracy_score(list(range(3)), [p for p,_ in preds])
    print('3-view prototype evaluation:', score, '(not an independent object-level validation)')
    np.savetxt(OUT / 'confusion_matrix.csv',
               confusion_matrix(list(range(3)), [p for p,_ in preds], labels=[0,1,2]),
               fmt='%d', delimiter=',')
    # Use a higher, collision-reducing transfer height and intermediate waypoints.
    # All movement remains controlled through joint commands, not object teleportation.
    if movie is not None:
        movie.start_recording(save_to_filename=str(OUT / 'waste_sorting_raw.mp4'), fps=10)
    hand = robot.get_link('hand')
    rigid = scene.sim.rigid_solver
    arm_dofs = np.arange(7)
    last_distance = None

    def vec(x):
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=float).reshape(-1)

    def item_position(item):
        return vec(item.get_pos())[:3]

    def hand_position():
        return vec(hand.get_pos())[:3]

    def move_to(p, steps=70, title=''):
        # Smooth joint-space interpolation avoids abrupt position commands.
        # Note: this does not guarantee collision-free joint trajectories.
        goal = vec(robot.inverse_kinematics(
            link=hand, pos=np.asarray(p, dtype=float),
            quat=np.array([0, 1, 0, 0])))[:7]
        begin = vec(robot.get_qpos())[:7]
        total = max(12, steps//3) if args.quick else max(steps, int(steps*args.speed))
        for t in range(total):
            alpha = (t + 1) / total
            blend = alpha*alpha*(3-2*alpha)
            robot.control_dofs_position(begin + (goal-begin)*blend, arm_dofs)
            scene.step()
        # Let position servo settle before activating suction constraint.
        for _ in range(3 if args.quick else 12):
            scene.step()
        actual = float(np.linalg.norm(hand_position()-np.asarray(p, dtype=float)))
        print(f'[ARM] {title:18s} target={np.round(p,3)} error={actual:.3f}m')
        return actual

    results = []
    try:
        # One object is handled at a time; recycle the single suction weld.
        for i, item in enumerate(items):
            predicted, confidence = preds[i]
            dst = np.array([DROP_X, DROP_Y[predicted]])
            obj_before = item_position(item)
            # Use the current measured pose, not an assumed original location.
            sx, sy, sz = map(float, obj_before)
            print(f'\n[ITEM {i+1}] {LABELS[i]} predicted={LABELS[predicted]} pos={obj_before}')
            # Approach from above, pause near object's top surface.
            # Paper is low and flat; bottle has a tall neck; can is intermediate.
            contact_z = {'plastic_bottle': 0.125,
                         'paper': 0.080,
                         'metal_can': 0.105}[LABELS[i]]
            move_to((sx, sy, 0.48), 65, 'above pickup')
            move_to((sx, sy, contact_z), 65, 'suction approach')
            gap = float(np.linalg.norm(hand_position()-item_position(item)))
            print(f'[GRASP] object-to-hand distance: {gap:.3f}m')
            # A weld fixes current relative pose. In this teaching prototype it
            # emulates a suction cup rather than friction-based finger grasp.
            obj_idx = item.get_link('object').idx
            rigid.add_weld_constraint(obj_idx, hand.idx)
            for _ in range(10 if args.quick else 25):
                scene.step()
            before_lift = item_position(item)
            # Raised transfer waypoint avoids the rim and the other bins.
            move_to((sx, sy, 0.53), 85, 'lift')
            after_lift = item_position(item)
            lift_gain = float(after_lift[2] - before_lift[2])
            print(f'[LIFT] delta z = {lift_gain:.3f}m')
            move_to((0.47, sy, 0.55), 65, 'clear pickup')
            move_to((0.47, float(dst[1]), 0.55), 85, 'cross at height')
            move_to((float(dst[0]), float(dst[1]), 0.55), 70, 'above bin center')
            # Drop from high above rim to avoid collision of robot with walls.
            # The item may remain offset from the hand because welding does not snap.
            held = item_position(item)
            print(f'[DROP] object before release={np.round(held,3)} bin={np.round(dst,3)}')
            rigid.delete_weld_constraint(obj_idx, hand.idx)
            # Do not keep arm in the falling object's path.
            move_to((float(dst[0]), float(dst[1]), 0.62), 40, 'retract')
            for _ in range(30 if args.quick else int(115*args.speed)):
                scene.step()
            final_pos = item_position(item)
            inside_xy = (abs(final_pos[0]-dst[0]) < BIN_W/2-WALL-0.008 and
                         abs(final_pos[1]-dst[1]) < BIN_D/2-WALL-0.008)
            # Final z prevents counting objects lying outside or on top of the rim.
            inside = inside_xy and (0.005 < final_pos[2] < BASE_H+BIN_H+0.10)
            success = bool(inside and predicted == i)
            print(f'[RESULT] final={np.round(final_pos,3)} in_target_bin={inside} '
                  f'class_correct={predicted==i} SUCCESS={success}')
            results.append([LABELS[i], LABELS[predicted], round(confidence,3),
                            round(gap,3), round(lift_gain,3),
                            *np.round(final_pos,3), int(inside), int(success)])
            # If a grasp or drop failed, record it; do not falsely report success.
        if not args.quick:
            for _ in range(100): scene.step()
    finally:
        if movie is not None:
            movie.stop_recording()
    with open(OUT/'robust_sorting_results.csv', 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['true_class','predicted_class','confidence','hand_object_gap_m',
                    'lift_delta_z_m','final_x','final_y','final_z','inside_bin','success'])
        w.writerows(results)
    if movie is not None:
        make_labeled_video(OUT / 'waste_sorting_raw.mp4', OUT / 'waste_sorting_robust_60plus.mp4')
        cap = cv2.VideoCapture(str(OUT / 'waste_sorting_robust_60plus.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        print(f'[VIDEO] duration={frames/fps:.1f}s (target 60-180s)')
    print(f'[SUMMARY] {sum(r[-1] for r in results)}/{len(results)} confirmed successful')
    print('Outputs:', OUT.resolve())

if __name__ == '__main__':
    main()
