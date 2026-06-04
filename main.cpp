#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <utility>
#include <queue>
#include <map>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <windows.h>

using namespace std;

#define VERIFY(result, error) \
    if(result != K4A_RESULT_SUCCEEDED) \
    { \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        Sleep(5000); \
        exit(1); \
    }

// 辅助函数：计算三个点的夹角（单位：度），顶点是b
float calculate_angle(k4a_float3_t a, k4a_float3_t b, k4a_float3_t c)
{
    // 向量 ba = a - b
    float bax = a.v[0] - b.v[0];
    float bay = a.v[1] - b.v[1];
    float baz = a.v[2] - b.v[2];
    
    // 向量 bc = c - b
    float bcx = c.v[0] - b.v[0];
    float bcy = c.v[1] - b.v[1];
    float bcz = c.v[2] - b.v[2];
    
    // 点积
    float dot = bax * bcx + bay * bcy + baz * bcz;
    // 模长
    float mag_ba = sqrt(bax*bax + bay*bay + baz*baz);
    float mag_bc = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
    
    // 计算cosθ，限制范围避免浮点误差
    float cos_theta = dot / (mag_ba * mag_bc);
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    
    // 转成角度
    return acos(cos_theta) * 180.0f / M_PI;
}

// 辅助函数：把深度相机的3D关节点，转成彩色图的2D像素坐标
cv::Point joint_to_pixel(const k4abt_joint_t &joint, const k4a_calibration_t &calib)
{
    k4a_float3_t pos3d = joint.position;
    k4a_float2_t pos2d;
    int valid = 0;
    k4a_calibration_3d_to_2d(
        &calib,
        &pos3d,
        K4A_CALIBRATION_TYPE_DEPTH,
        K4A_CALIBRATION_TYPE_COLOR,
        &pos2d,
        &valid
    );
    if (valid) {
        return cv::Point((int)pos2d.xy.x, (int)pos2d.xy.y);
    }
    return cv::Point(-1, -1);
}

int main()
{
    system("chcp 65001 > nul");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    float UPPER_BODY = 0.0f;
    float body_angle = 0.0f;
    k4a_device_t device = NULL;

    // 动作平滑：分开身体和手的，互不干扰
    queue<string> body_action_history;
    queue<string> hand_action_history;
    const int SMOOTH_FRAMES = 5;

    const uint32_t device_count = k4a_device_get_installed_count();
    if (device_count == 0)
    {
        cout << "Error: No K4A devices found!" << endl;
        Sleep(5000);
        exit(1);
    }
    cout << "Found " << device_count << " connected device." << endl;

    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");
    cout << "Done: open device." << endl;

    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.synchronized_images_only = true;

    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");
    cout << "Done: start camera." << endl;

    k4a_calibration_t sensor_calibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
           "Get depth camera calibration failed!");

    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");

    cv::Mat cv_rgbImage_with_alpha;
    cv::Mat cv_rgbImage_no_alpha;
    cv::Mat cv_depth;
    cv::Mat cv_depth_8U;
    int frame_count = 0;

    vector<pair<int, int>> bones = {
        {K4ABT_JOINT_PELVIS, K4ABT_JOINT_SPINE_NAVEL},
        {K4ABT_JOINT_SPINE_NAVEL, K4ABT_JOINT_SPINE_CHEST},
        {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_NECK},
        {K4ABT_JOINT_NECK, K4ABT_JOINT_HEAD},
        {K4ABT_JOINT_HEAD, K4ABT_JOINT_NOSE},
        {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_SHOULDER_LEFT},
        {K4ABT_JOINT_SHOULDER_LEFT, K4ABT_JOINT_ELBOW_LEFT},
        {K4ABT_JOINT_ELBOW_LEFT, K4ABT_JOINT_WRIST_LEFT},
        {K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_HAND_LEFT},
        {K4ABT_JOINT_HAND_LEFT, K4ABT_JOINT_HANDTIP_LEFT},
        {K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_THUMB_LEFT},
        {K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_SHOULDER_RIGHT},
        {K4ABT_JOINT_SHOULDER_RIGHT, K4ABT_JOINT_ELBOW_RIGHT},
        {K4ABT_JOINT_ELBOW_RIGHT, K4ABT_JOINT_WRIST_RIGHT},
        {K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_HAND_RIGHT},
        {K4ABT_JOINT_HAND_RIGHT, K4ABT_JOINT_HANDTIP_RIGHT},
        {K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_THUMB_RIGHT},
        {K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_LEFT},
        {K4ABT_JOINT_HIP_LEFT, K4ABT_JOINT_KNEE_LEFT},
        {K4ABT_JOINT_KNEE_LEFT, K4ABT_JOINT_ANKLE_LEFT},
        {K4ABT_JOINT_ANKLE_LEFT, K4ABT_JOINT_FOOT_LEFT},
        {K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_RIGHT},
        {K4ABT_JOINT_HIP_RIGHT, K4ABT_JOINT_KNEE_RIGHT},
        {K4ABT_JOINT_KNEE_RIGHT, K4ABT_JOINT_ANKLE_RIGHT},
        {K4ABT_JOINT_ANKLE_RIGHT, K4ABT_JOINT_FOOT_RIGHT}
    };

    while (true)
    {
        k4a_capture_t sensor_capture = NULL;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);

        k4a_image_t rgbImage = k4a_capture_get_color_image(sensor_capture);
        k4a_image_t depthImage = k4a_capture_get_depth_image(sensor_capture);

        if (rgbImage == NULL || depthImage == NULL)
        {
            printf("Warning: RGB/Depth image is null!\n");
            k4a_capture_release(sensor_capture);
            continue;
        }

        cv_rgbImage_with_alpha = cv::Mat(
            k4a_image_get_height_pixels(rgbImage),
            k4a_image_get_width_pixels(rgbImage),
            CV_8UC4,
            k4a_image_get_buffer(rgbImage)
        );
        cv::cvtColor(cv_rgbImage_with_alpha, cv_rgbImage_no_alpha, cv::COLOR_BGRA2BGR);

        cv_depth = cv::Mat(
            k4a_image_get_height_pixels(depthImage),
            k4a_image_get_width_pixels(depthImage),
            CV_16U,
            k4a_image_get_buffer(depthImage),
            k4a_image_get_stride_bytes(depthImage)
        );
        cv::normalize(cv_depth, cv_depth_8U, 0, 255, cv::NORM_MINMAX, CV_8U);

        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            frame_count++;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_capture_release(sensor_capture);

            if (queue_capture_result != K4A_WAIT_RESULT_SUCCEEDED)
            {
                printf("Error: Enqueue capture failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
            {
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                printf("Frame %d: Detected %zu bodies\n", frame_count, num_bodies);

                for (size_t i = 0; i < num_bodies; i++)
                {
                    k4abt_skeleton_t skeleton;
                    VERIFY(k4abt_frame_get_body_skeleton(body_frame, i, &skeleton), "Get skeleton failed!");
                    uint32_t body_id = k4abt_frame_get_body_id(body_frame, i);
                    printf("Body ID: %u\n", body_id);

                    // --------------------------
                    // 1. 骨骼绘制
                    // --------------------------
                    for (auto &bone : bones)
                    {
                        k4abt_joint_t j1 = skeleton.joints[bone.first];
                        k4abt_joint_t j2 = skeleton.joints[bone.second];
                        if (j1.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                            j2.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
                        {
                            cv::Point p1 = joint_to_pixel(j1, sensor_calibration);
                            cv::Point p2 = joint_to_pixel(j2, sensor_calibration);
                            if (p1.x > 0 && p1.y > 0 && p2.x > 0 && p2.y > 0)
                            {
                                cv::line(cv_rgbImage_no_alpha, p1, p2, cv::Scalar(0, 0, 255), 2);
                            }
                        }
                    }

                    for (int j = 0; j < K4ABT_JOINT_COUNT; j++)
                    {
                        k4abt_joint_t joint = skeleton.joints[j];
                        if (joint.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
                        {
                            cv::Point p = joint_to_pixel(joint, sensor_calibration);
                            if (p.x > 0 && p.y > 0)
                            {
                                cv::circle(cv_rgbImage_no_alpha, p, 4, cv::Scalar(0, 255, 0), -1);
                            }
                        }
                    }

                    // --------------------------
                    // 2. 身高计算
                    // --------------------------
                    k4abt_joint_t P_NECK = skeleton.joints[K4ABT_JOINT_NECK];
                    k4abt_joint_t P_PELVIS = skeleton.joints[K4ABT_JOINT_PELVIS];
                    k4abt_joint_t P_SHOULDER_R = skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT];
                    k4abt_joint_t P_HIP_R = skeleton.joints[K4ABT_JOINT_HIP_RIGHT];
                    k4abt_joint_t P_KNEE_R = skeleton.joints[K4ABT_JOINT_KNEE_RIGHT];
                    k4abt_joint_t P_SHOULDER_L = skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT];
                    k4abt_joint_t P_HIP_L = skeleton.joints[K4ABT_JOINT_HIP_LEFT];
                    k4abt_joint_t P_KNEE_L = skeleton.joints[K4ABT_JOINT_KNEE_LEFT];
                    k4abt_joint_t P_WRIST_L = skeleton.joints[K4ABT_JOINT_WRIST_LEFT];
                    k4abt_joint_t P_WRIST_R = skeleton.joints[K4ABT_JOINT_WRIST_RIGHT];
                    k4abt_joint_t P_ANKLE_L = skeleton.joints[K4ABT_JOINT_ANKLE_LEFT];
                    k4abt_joint_t P_ANKLE_R = skeleton.joints[K4ABT_JOINT_ANKLE_RIGHT];
                    k4abt_joint_t P_NOSE = skeleton.joints[K4ABT_JOINT_NOSE];

                    UPPER_BODY = sqrt(
                        pow(P_NECK.position.v[0] - P_PELVIS.position.v[0], 2) +
                        pow(P_NECK.position.v[1] - P_PELVIS.position.v[1], 2) +
                        pow(P_NECK.position.v[2] - P_PELVIS.position.v[2], 2)
                    );
                    float height_mm = UPPER_BODY * (1770.0f / 518.0f);
                    float height_cm = height_mm / 10.0f;

                    // --------------------------
                    // 3. 分开判断：手部动作 和 躯干动作，互不干扰！
                    // --------------------------
                    // --- 第一步：判断手部动作 ---
                    bool left_hand_over_head = P_WRIST_L.position.v[1] < P_NECK.position.v[1];
                    bool right_hand_over_head = P_WRIST_R.position.v[1] < P_NECK.position.v[1];
                    bool left_hand_reach = P_WRIST_L.position.v[2] < P_SHOULDER_L.position.v[2] - 100;
                    bool right_hand_reach = P_WRIST_R.position.v[2] < P_SHOULDER_R.position.v[2] - 100;

                    string current_hand_action = "Idle"; // 默认手没动作
                    if (left_hand_over_head && right_hand_over_head) {
                        current_hand_action = "Waving(Both)";
                    } else if (left_hand_over_head) {
                        current_hand_action = "Waving(Left)";
                    } else if (right_hand_over_head) {
                        current_hand_action = "Waving(Right)";
                    } else if (left_hand_reach && right_hand_reach) {
                        current_hand_action = "Reaching(Both)";
                    } else if (left_hand_reach) {
                        current_hand_action = "Reaching(Left)";
                    } else if (right_hand_reach) {
                        current_hand_action = "Reaching(Right)";
                    }

                    // --- 第二步：判断躯干动作！用膝盖角度，通用准确！ ---
                    string current_body_action = "Standing";
                    // 计算左右膝盖的角度
                    float left_knee_angle = calculate_angle(P_HIP_L.position, P_KNEE_L.position, P_ANKLE_L.position);
                    float right_knee_angle = calculate_angle(P_HIP_R.position, P_KNEE_R.position, P_ANKLE_R.position);
                    float avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0f;
                    
                    printf("Knee Angle: Left=%.1f, Right=%.1f, Avg=%.1f\n", left_knee_angle, right_knee_angle, avg_knee_angle);
                    
                    // 根据角度判断：
                    // 1. 角度>140°：腿几乎伸直 → 站着
                    // 2. 80°<角度<=140°：腿弯了一半 → 坐着
                    // 3. 角度<=80°：腿弯到底了 → 蹲着
                    if (avg_knee_angle <= 80.0f) {
                        current_body_action = "Crouching";
                    } else if (avg_knee_angle <= 140.0f) {
                        current_body_action = "Sitting";
                    } else {
                        current_body_action = "Standing";
                    }

                    // --------------------------
                    // 4. 分开平滑两个动作
                    // --------------------------
                    // 身体动作平滑
                    body_action_history.push(current_body_action);
                    if (body_action_history.size() > SMOOTH_FRAMES) body_action_history.pop();
                    map<string, int> body_count;
                    string final_body_action = current_body_action;
                    int max_body_cnt = 0;
                    queue<string> tmp_body = body_action_history;
                    while (!tmp_body.empty()) {
                        string a = tmp_body.front(); tmp_body.pop();
                        if (++body_count[a] > max_body_cnt) {
                            max_body_cnt = body_count[a];
                            final_body_action = a;
                        }
                    }
                    // 手部动作平滑
                    hand_action_history.push(current_hand_action);
                    if (hand_action_history.size() > SMOOTH_FRAMES) hand_action_history.pop();
                    map<string, int> hand_count;
                    string final_hand_action = current_hand_action;
                    int max_hand_cnt = 0;
                    queue<string> tmp_hand = hand_action_history;
                    while (!tmp_hand.empty()) {
                        string a = tmp_hand.front(); tmp_hand.pop();
                        if (++hand_count[a] > max_hand_cnt) {
                            max_hand_cnt = hand_count[a];
                            final_hand_action = a;
                        }
                    }

                    // --------------------------
                    // 5. 头部左边画红色文字：同时显示两个动作！
                    // 格式：躯干动作 | 手部动作 | 身高
                    // --------------------------
                    cv::Point nose_p = joint_to_pixel(P_NOSE, sensor_calibration);
                    if (nose_p.x > 0 && nose_p.y > 0) {
                        char text[128];
                        sprintf(text, "%s | %s | %.1f cm", final_body_action.c_str(), final_hand_action.c_str(), height_cm);
                        cv::putText(
                            cv_rgbImage_no_alpha,
                            text,
                            cv::Point(nose_p.x - 220, nose_p.y), // 左移多一点，放下长文字
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.9, // 字体稍微缩小一点，避免超出屏幕
                            cv::Scalar(0, 0, 255),
                            2
                        );
                    }

                    // --------------------------
                    // 终端输出
                    // --------------------------
                    printf("Neck: %.2f, %.2f, %.2f\n", P_NECK.position.v[0], P_NECK.position.v[1], P_NECK.position.v[2]);
                    printf("Pelvis: %.2f, %.2f, %.2f\n", P_PELVIS.position.v[0], P_PELVIS.position.v[1], P_PELVIS.position.v[2]);
                    printf("Estimated Height: %.1f cm\n", height_cm);
                    printf("Body Action: %s, Hand Action: %s\n", final_body_action.c_str(), final_hand_action.c_str());
                    printf("----------------------------------------\n");
                }
                k4abt_frame_release(body_frame);
            }
        }

        cv::imshow("Color (Skeleton)", cv_rgbImage_no_alpha);
        cv::imshow("Depth", cv_depth_8U);

        k4a_image_release(rgbImage);
        k4a_image_release(depthImage);

        if (cv::waitKey(1) == 27)
        {
            printf("ESC pressed, exit.\n");
            break;
        }
    }

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    k4a_device_stop_cameras(device);
    k4a_device_close(device);
    cv::destroyAllWindows();

    return 0;
}