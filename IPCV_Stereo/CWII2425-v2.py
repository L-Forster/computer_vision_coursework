
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication

    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix

    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__':

        ####################################
        ### Take command line arguments ####
        ####################################

        parser = argparse.ArgumentParser()
        parser.add_argument('--num', dest='num', type=int, default=6,
                            help='number of spheres')
        parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10,
                            help='min sphere  radius x10')
        parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16,
                            help='max sphere  radius x10')
        parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4,
                           help='min sphere  separation')
        parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8,
                           help='max sphere  separation')
        parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                            help='open up another visualiser to visualise centres')
        parser.add_argument('--coords', dest='bCoords', action='store_true')

        args = parser.parse_args()

        if args.num<=0:
            print('invalidnumber of spheres')
            exit()

        if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
            print('invalid max and min sphere radii')
            exit()

        if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
            print('invalid max and min sphere separation')
            exit()

        ####################################
        #### Setup objects in the scene ####
        ####################################

        # create plane to hold all spheres
        h, w = 24, 12
        # place the support plane on the x-z plane
        box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
        box_H=np.array(
                     [[1, 0, 0, -h/2],
                      [0, 1, 0, -0.05],
                      [0, 0, 1, -w/2],
                      [0, 0, 0, 1]]
                    )
        box_rgb = [0.7, 0.7, 0.7]
        name_list = ['plane']
        mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

        # create spheres
        prev_loc = []
        GT_cents, GT_rads = [], []
        for i in range(args.num):
            # add sphere name
            name_list.append(f'sphere_{i}')

            # create sphere with random radius
            size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
            sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
            mesh_list.append(sph_mesh)
            RGB_list.append([0., 0.5, 0.5])

            # create random sphere location
            step = random.randrange(int(args.sph_sep_min),int(args.sph_sep_max),1)
            x = random.randrange(int(-h/2+2), int(h/2-2), step)
            z = random.randrange(int(-w/2+2), int(w/2-2), step)
            while check_dup_locations(x, z, prev_loc):
                x = random.randrange(int(-h/2+2), int(h/2-2), step)
                z = random.randrange(int(-w/2+2), int(w/2-2), step)
            prev_loc.append((x, z))

            GT_cents.append(np.array([x, size, z, 1.]))
            GT_rads.append(size)
            sph_H = np.array(
                        [[1, 0, 0, x],
                         [0, 1, 0, size],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]]
                    )
            H_list.append(sph_H)

        # arrange plane and sphere in the space
        obj_meshes = []
        for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
            # apply location
            mesh.vertices = o3d.utility.Vector3dVector(
                transform_points(np.asarray(mesh.vertices), H)
            )
            # paint meshes in uniform colours here
            mesh.paint_uniform_color(rgb)
            mesh.compute_vertex_normals()
            obj_meshes.append(mesh)

        # add optional coordinate system
        if args.bCoords:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
            obj_meshes = obj_meshes+[coord_frame]
            RGB_list.append([1., 1., 1.])
            name_list.append('coords')

        ###################################
        #### Setup camera orientations ####
        ###################################

        # set camera pose (world to camera)
        # # camera init
        # # placed at the world origin, and looking at z-positive direction,
        # # x-positive to right, y-positive to down
        # H_init = np.eye(4)
        # print(H_init)

        # camera_0 (world to camera)
        theta = np.pi * (45*5+random.uniform(-5, 5))/180.
        # theta = np.pi * (225)/180.
        H0_wc = np.array(
                    [[1,            0,              0,  0],
                    [0, np.cos(theta), -np.sin(theta),  0],
                    [0, np.sin(theta),  np.cos(theta), 20],
                    [0, 0, 0, 1]]
                )

        # camera_1 (world to camera)
        theta = np.pi * (80+random.uniform(-10, 10))/180.
        # theta = np.pi * (90)/180.
        H1_0 = np.array(
                    [[np.cos(theta),  0, np.sin(theta), 0],
                     [0,              1, 0,             0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]]
                )
        theta = np.pi * (45*5+random.uniform(-5, 5))/180.
        # theta = np.pi * (225)/180.
        H1_1 = np.array(
                    [[1, 0,            0,              0],
                    [0, np.cos(theta), -np.sin(theta), -4],
                    [0, np.sin(theta), np.cos(theta),  20],
                    [0, 0, 0, 1]]
                )
        H1_wc = np.matmul(H1_1, H1_0)
        render_list = [(H0_wc, 'view0.png', 'depth0.png'),
                       (H1_wc, 'view1.png', 'depth1.png')]

    #####################################################
        # NOTE: This section relates to rendering scenes in Open3D, details are not
        # critical to understanding the lab, but feel free to read Open3D docs
        # to understand how it works.

        # set up camera intrinsic matrix needed for rendering in Open3D
        img_width=640
        img_height=480
        f=415 # focal length
        # image centre in pixel coordinates
        ox=img_width/2-0.5
        oy=img_height/2-0.5
        K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)
        # Rendering RGB-D frames given camera poses
        # create visualiser and get rendered views
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = K
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=img_width, height=img_height, left=0, top=0)
        for m in obj_meshes:
            vis.add_geometry(m)
        ctr = vis.get_view_control()
        for (H_wc, name, dname) in render_list:
            cam.extrinsic = H_wc
            ctr.convert_from_pinhole_camera_parameters(cam,True)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(name, True)
            vis.capture_depth_image(dname, True)
        vis.run()
        vis.destroy_window()
    ##################################################
        # load in the images for post processings
        img0 = cv2.imread('view0.png', -1)
        dep0 = cv2.imread('depth0.png', -1)
        img1 = cv2.imread('view1.png', -1)
        dep1 = cv2.imread('depth1.png', -1)
        print(GT_cents)
        # visualise sphere centres
        pcd_GTcents = o3d.geometry.PointCloud()
        pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
        pcd_GTcents.paint_uniform_color([1., 0., 0.])
        if args.bCentre:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=480, left=0, top=0)
            for m in [obj_meshes[0], pcd_GTcents]:
                vis.add_geometry(m)
            vis.run()
            vis.destroy_window()


        ###################################
        '''
        Task 3: Circle detection
        Hint: use cv2.HoughCircles() for circle detection.
        https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    
        Write your code here
        '''
        ###################################
        #pre-process images
        total_circle_centres = []
        total_circle_radii = []
        image_names = ["depth0.png", "view0.png", "depth1.png", "view1.png"]
        for img_name in image_names:
            circle_centres = []
            circle_radii = []
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            cimg = cv2.imread(img_name, cv2.COLOR_GRAY2BGR)
            img = cv2.GaussianBlur(img, (7, 7), sigmaX=1.5, sigmaY=1.5)

            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.5, 30,param1=100, param2=20, minRadius=10, maxRadius=70)
            print(circles)

            if circles is not None:
                for i in circles[0, :]:
                    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                    if len(i)!=0:
                        # priknecnt(i)
                        circle_radii.append(i[2])
                        circle_centres.append([i[0],i[1]])
                print(circle_centres)
            total_circle_centres.append(circle_centres)
            total_circle_radii.append(circle_radii)
            cv2.imwrite(img_name, cimg)

        # asserting that the two views we are comparing have the same size
        # assert len(total_circle_centres[1]) == len(total_circle_centres[3])

        ###################################
        '''
        Task 4: Epipolar line
        Hint: Compute Essential & Fundamental Matrix
              Draw lines with cv2.line() function
        https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
        
        Write your code here
        '''
        ###################################
        def run_all(H0_wc, H1_wc):
            # for view 1
            view0_centres = total_circle_centres[1]
            view1_radii = total_circle_radii[3]
            # print(total_circle_centres[1])


            # Difference view1 to view0: Right column of H0_WC - Right column of H1_WC

            H0_cw = np.linalg.inv(H0_wc) # from c0 to World

            H_01 = H1_wc @ H0_cw  # from C0 to C1
            translate = np.array(H_01[:3,3]) #translation part of 0 to 1
            print(translate)
            rotation = H_01[:3,:3]
            print(rotation)
            R = rotation
            T = translate
            # skew-symmetric matrix
            S = np.array([
                [0, -T[2], T[1]],
                [T[2], 0, -T[0]],
                [-T[1], T[0], 0]
            ])

            E = S @ R
            print("E=",E)

            F = np.linalg.inv(K.intrinsic_matrix.T) @ E @ np.linalg.inv(K.intrinsic_matrix)

            image = cv2.imread("view1.png", -1)
            print("F=",F)
            view0_centres = np.column_stack((view0_centres, np.ones(np.array(view0_centres).shape[0])))

            # using given formula to calculate u
            lines = F @ view0_centres.T
            lines = lines.T
            norms = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2).reshape(-1, 1)
            lines = lines / norms
            # print(lines)
            # print(image.shape)
            lines = lines.reshape(-1,3)
            # print(lines)
            epipolar_lines = []

            r,c,_ = image.shape
            for i in range(len(lines)):
                # start and end points
                x0,y0 = map(int, [0, -lines[i][2]/lines[i][1] ])
                x1,y1 = map(int, [c, -(lines[i][2]+lines[i][0]*c)/lines[i][1] ])
                img1 = cv2.line(image, (x0,y0), (x1,y1), (0,255,0),1)
                # image_with_text = cv2.putText(image,str(i),org=((x0 + x1) // 2, (y0 + y1) // 2),
                #     fontFace=cv2.FONT_HERSHEY_PLAIN,
                #     fontScale=1,
                #     color=(0, 0, 255),
                #     thickness=1
                # )
                epipolar_lines.append([(x0,y0),(x1,y1)])

            # calculate endpoints
            cv2.imwrite("epipolar_lines.png", image)
            #display the line

            ###################################
            '''
            Task 5: Find correspondences
        
            Write your code here
            '''
            ###################################
            # correspondences

            closest_points = []
            view1_centres = np.array(total_circle_centres[3])
            # print("shape0",view1_centres.shape)

            # print(view0_centres)
            # print(view1_centres)
            updated_view1_centres = np.zeros_like(view1_centres)
            updated_view1_radii = np.zeros_like(view1_radii)
            # print("shape: ",updated_view1_centres.shape)
            for i in range(len(view1_centres)):
                min_dist = float('inf')
                # calculating the closes line for each centre
                for j in range(len(lines)):
                    a, b, c = lines[j]
                    distance = np.abs(a * view1_centres[i][0] + b * view1_centres[i][1] + c) / np.sqrt(a ** 2 + b ** 2)
                    # print(lines[j],a,b,view1_centres[i][:2], distance)
                    if distance < min_dist:
                        min_dist = distance
                        min_index = j
                        # print("lowest_distance", j)
                # print(view1_centres[i], min_index)
                # assigning the centres to position based on position of lines
                if min_index < len(updated_view1_centres):
                    updated_view1_centres[min_index] = view1_centres[i]
                if min_index < len(updated_view1_radii):
                    updated_view1_radii[min_index] = view1_radii[i]

            view1_centres = updated_view1_centres
            view1_radii = updated_view1_radii
            # print("Start:")
            # print(view0_centres)
            # print(view1_centres)

            ###################################
            '''
            Task 6: 3-D locations of sphere centres
        
            Write your code here
            '''
            ###################################
            # define L = 0 and R = 1

            view1_centres = np.array(view1_centres)
            view0_centres = view0_centres[:,:2]

            # transform into image coords
            # view1_centres = np.column_stack((view1_centres, np.ones(np.array(view1_centres).shape[0])))
            view0_centres = np.hstack((view0_centres, np.full((view0_centres.shape[0], 1), 1)))
            view1_centres = np.hstack((view1_centres, np.full((view1_centres.shape[0], 1), 1)))
            print(view0_centres)
            print(view1_centres)
            t1 = view0_centres

            # M : transformation applied to pixel coordinates to get to image coordinates
            M = np.linalg.inv(K.intrinsic_matrix)


            # print(M)
            p_hats = []
            view1_centres = view1_centres.astype(float)
            # compute 3d coords in view 0
            H1_cw = np.linalg.inv(H1_wc)

            origin = np.array([0,0,0,1])
            print(origin, H0_cw, H1_cw)
            origin_0 = H0_cw @ origin
            origin_1 = H1_cw @ origin
            # print("Origins", origin_0, origin_1)

            # converting coordinates into image space
            view1_centres = np.atleast_2d(view1_centres)
            print(view1_centres)
            for i in range(min(len(view1_centres),len(view0_centres))):
                # print(view0_centres[i])
                # print(view1_centres[i])
                view0_centres[i] = M @ view0_centres[i]
                view1_centres[i] = M @ view1_centres[i]
            view0_centres = view0_centres[:len(view1_centres),:]
            # add dummy dimension
            view0_centres = np.hstack((view0_centres, np.full((view0_centres.shape[0], 1), 1)))
            view1_centres = np.hstack((view1_centres, np.full((view1_centres.shape[0], 1), 1)))

            for i in range(len(view0_centres)):
                # calculate points in world
                p0_w = H0_cw @ view0_centres[i]
                p1_w = H1_cw @ view1_centres[i]

                # print(p0_w, p1_w)
                origin_0 = origin_0[:3]
                origin_1 = origin_1[:3]
                v0 = p0_w[:3] - origin_0
                v1 = p1_w[:3] - origin_1
                H = np.vstack([v0, -v1, np.cross(v0, v1)]).T
                T = origin_1 - origin_0
                a, b, c = np.linalg.inv(H) @ T
                # final 3d point calculation
                p_hat = (a * v0 + origin_0 + b * v1 + origin_1) / 2
                p_hats.append(p_hat)

            print("Ground Truths: ",GT_cents)

            ###################################
            '''
            Task 7: Evaluate and Display the centres
        
            Write your code here
            '''
            ###################################

            sphere_coords = np.array(p_hats)
            sphere_coords = sphere_coords.astype(np.float64)
            if sphere_coords.ndim == 1:
                sphere_coords = sphere_coords.reshape(1, -1)

            GT_cents_np = np.array(GT_cents)[:,:3]
            correspondences = []

            # this might assign multiple spheres to one GT
            for pred in sphere_coords:
                distances = np.linalg.norm(GT_cents_np - pred, axis=1)
                print(distances)
                closest_index = np.argmin(distances)
                correspondences.append(closest_index)
            print("GT Coords", GT_cents_np)
            print("3D Sphere coords", sphere_coords)
            print(correspondences)

            #calculate errors as the error between correspondences
            errors = []
            for i in range(len(correspondences)):
                i1 = correspondences[i]
                error = np.linalg.norm(sphere_coords[i] - GT_cents_np[i1])
                errors.append(error)


            print("Error between correspondences", errors)
            # ground truths
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(GT_cents_np[:,:3])
            pcd_gt.paint_uniform_color([0.0, 1.0, 0.0])

            # sphere estimates
            pcd_sphere = o3d.geometry.PointCloud()
            pcd_sphere.points = o3d.utility.Vector3dVector(sphere_coords)
            pcd_sphere.paint_uniform_color([1.0, 0.0, 0.0])

            geometries = [obj_meshes[0], pcd_gt, pcd_sphere]

            vis = o3d.visualization.Visualizer()
            vis.create_window()

            for geometry in geometries:
                vis.add_geometry(geometry)

            view_ctl = vis.get_view_control()
            view_ctl.set_up([0, -1, 0])
            view_ctl.set_front([0, 0, -1])
            view_ctl.set_lookat([0, 0, 0])
            view_ctl.set_zoom(0.8)

            vis.run()
            vis.destroy_window()


            ###################################
            '''
            Task 8: 3-D radius of spheres
        
            Write your code here
            '''
            ###################################
            # calculating the depth of the centres
            depths = []
            for i in range(len(sphere_coords)):
                temp = - sphere_coords[i] + origin_1[:3]
                depths.append(np.linalg.norm(temp))

            # calculating the radii from the depth
            print(view1_radii, GT_rads)
            for i in range(min(len(view1_radii),len(depths))): # preventing mismatch
                temp_rad = float(view1_radii[i] * (depths[i]/ f))
                temp_arr = np.array([0, 0, temp_rad])
                radius = np.linalg.norm(temp_rad)
                # if triangulation failed due to errors in hough circles
                if radius <= 0:
                    # set it to a default value
                    view1_radii[i] = 1.2
                else:
                    view1_radii[i] = np.linalg.norm(temp_rad)


            print(view1_radii)
            ###################################
            '''
            Task 9: Display the spheres
            Display the estimated spheres alongside the ground truth spheres and compute the
            error in the radius estimates. Think carefully about how you display the estimated and
            ground truth spheres so as to allow comparison, since comparison may be difficult if
            you display the estimates as solid spheres.

            
            Write your code here:
            '''
            ###################################

            # radius error
            print("Radius error",  np.abs(view1_radii - GT_rads[:len(view1_radii)]))

            # displaying the spheres in 3D
            def create_sphere_at(centre, radius, color):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
                sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
                sphere.translate(centre)
                sphere.paint_uniform_color(color)  #
                return sphere

            gt_spheres = []
            for centre, radius in zip(GT_cents_np[:, :3], GT_rads):
                gt_spheres.append(create_sphere_at(centre, radius, [0.0, 1.0, 0.0]))

            pred_spheres = []
            for centre, radius in zip(sphere_coords, view1_radii):
                pred_spheres.append(create_sphere_at(centre, radius, [1.0, 0.0, 0.0]))

            geometries = [obj_meshes[0]] + gt_spheres + pred_spheres

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().line_width = 2.0
            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.run()
            vis.destroy_window()


        run_all(H0_wc, H1_wc)
        ###################################
        '''
        Task 10: Investigate impact of noise added to relative pose
    
        Write your code here:
        '''
        ###################################

        noise_scalar = 1

        def apply_noise(H,rotation_scalar, translation_scalar):
            rotation = H[:3,:3]
            x_angle = np.random.normal(0,1) * rotation_scalar
            y_angle = np.random.normal(0,1) * rotation_scalar
            z_angle = np.random.normal(0,1) * rotation_scalar

            x_matrix = np.array([[1,0,0],
                                 [0,np.cos(x_angle),-np.sin(x_angle)],
                                 [0,np.sin(x_angle), np.cos(x_angle)]])

            y_matrix = np.array([[np.cos(y_angle),0,np.sin(y_angle)],
                                 [0,1,0],
                                 [-np.sin(y_angle),0, np.cos(y_angle)]])

            z_matrix = np.array([[np.cos(z_angle),-np.sin(z_angle),0],
                                 [np.sin(z_angle), np.cos(z_angle),0],
                                 [0,0,1]])

            rotation = rotation @ x_matrix @ y_matrix @ z_matrix

            translation = H[:3,3]
            translation += np.random.normal(0,1) * translation_scalar

            return rotation, translation
        r0_noise, t0_noise = apply_noise(H0_wc, 0.1, 0.5)
        H0_noise = np.eye(4)
        H0_noise[:3, :3] = r0_noise
        H0_noise[:3, 3] = t0_noise

        r1_noise, t1_noise = apply_noise(H1_wc, 0.1, 0.5)
        H1_noise = np.eye(4)
        H1_noise[:3, :3] = r1_noise
        H1_noise[:3, 3] = t1_noise

        run_all(H0_noise, H1_noise)