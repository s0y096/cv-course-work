#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <utility>
#include <queue>
#include <map>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <string>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <windows.h>

using namespace std;
using namespace std::chrono;

#define VERIFY(result, error) \
    if(result != K4A_RESULT_SUCCEEDED) \
    { \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        Sleep(5000); \
        exit(1); \
    }

// ---------- 多人追踪：持久化 ID 数据结构 ----------
struct TrackedBody {
    int tracked_id;                    // 我们分配的持久 ID
    uint32_t k4a_body_id;             // Kinect 原始 body_id
    k4a_float3_t pelvis_pos;          // 上一帧骨盆 3D 位置
    int last_seen_frame;              // 最后出现的帧号
    int age;                          // 已追踪帧数
};

// 全局追踪状态
static int next_tracked_id = 1;
static unordered_map<int, TrackedBody> tracked_bodies;  // tracked_id -> TrackedBody
const float TRACKING_DIST_THRESHOLD = 0.5f;  // 匹配距离阈值（米），500mm
const int MAX_MISSING_FRAMES = 30;           // 丢失超过 30 帧则删除

// 辅助函数：计算两个 3D 点的欧氏距离（单位：毫米）
float dist_3d(const k4a_float3_t &a, const k4a_float3_t &b) {
    float dx = a.v[0] - b.v[0];
    float dy = a.v[1] - b.v[1];
    float dz = a.v[2] - b.v[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// 辅助函数：计算三个点的夹角（单位：度），顶点是b
float calculate_angle(k4a_float3_t a, k4a_float3_t b, k4a_float3_t c) {
    float bax = a.v[0] - b.v[0];
    float bay = a.v[1] - b.v[1];
    float baz = a.v[2] - b.v[2];
    
    float bcx = c.v[0] - b.v[0];
    float bcy = c.v[1] - b.v[1];
    float bcz = c.v[2] - b.v[2];
    
    float dot = bax * bcx + bay * bcy + baz * bcz;
    float mag_ba = sqrt(bax*bax + bay*bay + baz*baz);
    float mag_bc = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
    
    float cos_theta = dot / (mag_ba * mag_bc);
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    
    return acos(cos_theta) * 180.0f / M_PI;
}

// 辅助函数：把深度相机的3D关节点，转成彩色图的2D像素坐标
cv::Point joint_to_pixel(const k4abt_joint_t &joint, const k4a_calibration_t &calib) {
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

// ---------- 数据可视化侧边栏：返回独立的 Mat，拼接到视频右侧 ----------
cv::Mat create_sidebar(int video_height, int frame_count, float fps, size_t num_bodies,
                       const vector<int> &body_ids, const vector<float> &heights,
                       const vector<float> &knee_angles, const vector<string> &body_actions,
                       const vector<string> &hand_actions) {
    const int SIDEBAR_W = 340;
    const int line_height = 22;
    const int header_h = 30;
    const int margin = 12;

    cv::Mat sidebar(video_height, SIDEBAR_W, CV_8UC3, cv::Scalar(30, 30, 30));

    int y = margin + 20;

    // ---- 标题 ----
    char title[128];
    sprintf(title, "INFO | FPS: %.1f | Frame: %d | Bodies: %zu", fps, frame_count, num_bodies);
    cv::putText(sidebar, title, cv::Point(margin, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    y += line_height + 6;

    // 分隔线
    cv::line(sidebar, cv::Point(margin, y), cv::Point(SIDEBAR_W - margin, y),
             cv::Scalar(0, 180, 0), 2);
    y += 12;

    if (num_bodies == 0) {
        cv::putText(sidebar, "No body detected", cv::Point(margin, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);
        return sidebar;
    }

    for (size_t i = 0; i < body_ids.size(); i++) {
        char buf[128];

        // 人物卡片背景
        int card_top = y - 4;
        int card_h = 6 * line_height + 8;
        cv::rectangle(sidebar, cv::Rect(margin - 4, card_top, SIDEBAR_W - 2 * margin + 8, card_h),
                      cv::Scalar(50, 50, 50), cv::FILLED);
        cv::rectangle(sidebar, cv::Rect(margin - 4, card_top, SIDEBAR_W - 2 * margin + 8, card_h),
                      cv::Scalar(80, 80, 80), 1);

        // 人物 ID + 身高
        sprintf(buf, "Body #%d  |  Height: %.1f cm", body_ids[i], heights[i]);
        cv::putText(sidebar, buf, cv::Point(margin + 4, y + line_height - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        y += line_height;

        // 膝盖角度
        sprintf(buf, "  Knee Angle: %.1f deg", knee_angles[i]);
        cv::putText(sidebar, buf, cv::Point(margin + 4, y + line_height - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        y += line_height;

        // 躯干动作
        cv::Scalar body_color = (body_actions[i] == "Standing") ? cv::Scalar(0, 255, 0) :
                                (body_actions[i] == "Sitting")  ? cv::Scalar(255, 200, 0) :
                                                                  cv::Scalar(255, 100, 0);
        sprintf(buf, "  Body: %s", body_actions[i].c_str());
        cv::putText(sidebar, buf, cv::Point(margin + 4, y + line_height - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, body_color, 1);
        y += line_height;

        // 手部动作
        cv::Scalar hand_color = (hand_actions[i] == "Idle") ? cv::Scalar(180, 180, 180) :
                                                               cv::Scalar(0, 255, 255);
        sprintf(buf, "  Hand: %s", hand_actions[i].c_str());
        cv::putText(sidebar, buf, cv::Point(margin + 4, y + line_height - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1);
        y += line_height;

        // 置信度
        sprintf(buf, "  Joint Confidence: OK");
        cv::putText(sidebar, buf, cv::Point(margin + 4, y + line_height - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 150, 150), 1);
        y += line_height + 10;  // 间距
    }

    return sidebar;
}

int main() {
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

    // ---------- FPS 计时 ----------
    auto last_time = steady_clock::now();
    float fps = 0.0f;
    int frame_count = 0;

    const uint32_t device_count = k4a_device_get_installed_count();
    if (device_count == 0) {
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

    while (true) {
        // ---------- FPS 计算 ----------
        auto current_time = steady_clock::now();
        float elapsed = duration<float>(current_time - last_time).count();
        last_time = current_time;
        if (elapsed > 0.001f) {
            fps = 0.9f * fps + 0.1f * (1.0f / elapsed);  // 平滑 FPS
        }

        k4a_capture_t sensor_capture = NULL;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);

        k4a_image_t rgbImage = k4a_capture_get_color_image(sensor_capture);
        k4a_image_t depthImage = k4a_capture_get_depth_image(sensor_capture);

        if (rgbImage == NULL || depthImage == NULL) {
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

        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED) {
            frame_count++;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_capture_release(sensor_capture);

            if (queue_capture_result != K4A_WAIT_RESULT_SUCCEEDED) {
                printf("Error: Enqueue capture failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);

            // ---------- 面板数据收集 ----------
            vector<int> panel_body_ids;
            vector<float> panel_heights;
            vector<float> panel_knee_angles;
            vector<string> panel_body_actions;
            vector<string> panel_hand_actions;

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED) {
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                printf("Frame %d: Detected %zu bodies\n", frame_count, num_bodies);

                // ---------- 多人追踪：收集当前帧所有骨盆位置 ----------
                struct DetectedBody {
                    int frame_index;
                    k4a_float3_t pelvis;
                    uint32_t k4a_id;
                };
                vector<DetectedBody> detections;

                for (size_t i = 0; i < num_bodies; i++) {
                    k4abt_skeleton_t skeleton;
                    VERIFY(k4abt_frame_get_body_skeleton(body_frame, i, &skeleton), "Get skeleton failed!");
                    uint32_t body_id = k4abt_frame_get_body_id(body_frame, i);
                    k4abt_joint_t pelvis_joint = skeleton.joints[K4ABT_JOINT_PELVIS];
                    detections.push_back({(int)i, pelvis_joint.position, body_id});
                }

                // ---------- 多人追踪：匹配或创建 ID ----------
                vector<int> assigned_ids(detections.size(), -1);
                vector<bool> tracked_used(tracked_bodies.size(), false);

                // 构建 tracked body 列表用于匹配
                vector<pair<int, TrackedBody*>> track_list;
                for (auto &kv : tracked_bodies) {
                    track_list.push_back({kv.first, &kv.second});
                }

                // 对每个检测，找最近的 tracked body
                for (size_t d = 0; d < detections.size(); d++) {
                    float best_dist = TRACKING_DIST_THRESHOLD * 1000.0f;  // 转换为 mm
                    int best_tracked_id = -1;

                    for (size_t t = 0; t < track_list.size(); t++) {
                        if (tracked_used[t]) continue;
                        float d_mm = dist_3d(detections[d].pelvis, track_list[t].second->pelvis_pos);
                        if (d_mm < best_dist) {
                            best_dist = d_mm;
                            best_tracked_id = track_list[t].first;
                        }
                    }

                    if (best_tracked_id >= 0) {
                        assigned_ids[d] = best_tracked_id;
                        // 标记此 tracked body 已被使用
                        for (size_t t = 0; t < track_list.size(); t++) {
                            if (track_list[t].first == best_tracked_id) {
                                tracked_used[t] = true;
                                break;
                            }
                        }
                    }
                }

                // 为未匹配的检测分配新 ID
                for (size_t d = 0; d < detections.size(); d++) {
                    if (assigned_ids[d] < 0) {
                        assigned_ids[d] = next_tracked_id++;
                    }
                }

                // 更新 tracked_bodies
                for (size_t d = 0; d < detections.size(); d++) {
                    int tid = assigned_ids[d];
                    if (tracked_bodies.find(tid) == tracked_bodies.end()) {
                        TrackedBody tb;
                        tb.tracked_id = tid;
                        tb.age = 0;
                        tracked_bodies[tid] = tb;
                    }
                    tracked_bodies[tid].k4a_body_id = detections[d].k4a_id;
                    tracked_bodies[tid].pelvis_pos = detections[d].pelvis;
                    tracked_bodies[tid].last_seen_frame = frame_count;
                    tracked_bodies[tid].age++;
                }

                // 清理超时未出现的 tracked body
                vector<int> to_remove;
                for (auto &kv : tracked_bodies) {
                    if (frame_count - kv.second.last_seen_frame > MAX_MISSING_FRAMES) {
                        to_remove.push_back(kv.first);
                    }
                }
                for (int rid : to_remove) {
                    tracked_bodies.erase(rid);
                }

                // ---------- 绘制 & 动作识别 ----------
                for (size_t i = 0; i < num_bodies; i++) {
                    k4abt_skeleton_t skeleton;
                    VERIFY(k4abt_frame_get_body_skeleton(body_frame, i, &skeleton), "Get skeleton failed!");
                    uint32_t body_id = k4abt_frame_get_body_id(body_frame, i);
                    int persistent_id = assigned_ids[i];

                    printf("Body ID: %u (Tracked #%d)\n", body_id, persistent_id);

                    // --------------------------
                    // 1. 骨骼绘制
                    // --------------------------
                    for (auto &bone : bones) {
                        k4abt_joint_t j1 = skeleton.joints[bone.first];
                        k4abt_joint_t j2 = skeleton.joints[bone.second];
                        if (j1.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                            j2.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW) {
                            cv::Point p1 = joint_to_pixel(j1, sensor_calibration);
                            cv::Point p2 = joint_to_pixel(j2, sensor_calibration);
                            if (p1.x > 0 && p1.y > 0 && p2.x > 0 && p2.y > 0) {
                                cv::line(cv_rgbImage_no_alpha, p1, p2, cv::Scalar(0, 0, 255), 2);
                            }
                        }
                    }

                    for (int j = 0; j < K4ABT_JOINT_COUNT; j++) {
                        k4abt_joint_t joint = skeleton.joints[j];
                        if (joint.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW) {
                            cv::Point p = joint_to_pixel(joint, sensor_calibration);
                            if (p.x > 0 && p.y > 0) {
                                cv::circle(cv_rgbImage_no_alpha, p, 4, cv::Scalar(0, 255, 0), -1);
                            }
                        }
                    }

                    // --------------------------
                    // 2. 更精确的身高估算：
                    //    身高 = Head Y 坐标 − 双足踝 Y 坐标平均值 + 补偿值
                    //    Y 轴在 Kinect 中向上为负，向下为正，所以是 Foot_Y − Head_Y
                    //    Head 关节 ≈ 头顶下方 2−4cm，Foot 关节 ≈ 脚踝 ≈ 脚底上方 8−10cm
                    //    补偿值 ≈ 头骨上方 + 脚底 = 约 10−14cm
                    //    通过 HYPER_CONSTANT 统一校准，用户可自行调参
                    // --------------------------
                    k4abt_joint_t P_NECK   = skeleton.joints[K4ABT_JOINT_NECK];
                    k4abt_joint_t P_PELVIS = skeleton.joints[K4ABT_JOINT_PELVIS];
                    k4abt_joint_t P_SHOULDER_R = skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT];
                    k4abt_joint_t P_HIP_R  = skeleton.joints[K4ABT_JOINT_HIP_RIGHT];
                    k4abt_joint_t P_KNEE_R = skeleton.joints[K4ABT_JOINT_KNEE_RIGHT];
                    k4abt_joint_t P_SHOULDER_L = skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT];
                    k4abt_joint_t P_HIP_L  = skeleton.joints[K4ABT_JOINT_HIP_LEFT];
                    k4abt_joint_t P_KNEE_L = skeleton.joints[K4ABT_JOINT_KNEE_LEFT];
                    k4abt_joint_t P_WRIST_L = skeleton.joints[K4ABT_JOINT_WRIST_LEFT];
                    k4abt_joint_t P_WRIST_R = skeleton.joints[K4ABT_JOINT_WRIST_RIGHT];
                    k4abt_joint_t P_ANKLE_L = skeleton.joints[K4ABT_JOINT_ANKLE_LEFT];
                    k4abt_joint_t P_ANKLE_R = skeleton.joints[K4ABT_JOINT_ANKLE_RIGHT];
                    k4abt_joint_t P_NOSE   = skeleton.joints[K4ABT_JOINT_NOSE];
                    k4abt_joint_t P_HEAD   = skeleton.joints[K4ABT_JOINT_HEAD];

                    // **** 可调整的校准常数 ****
                    // Head 关节 ≈ 眉心/眼高，距头顶约 10-12cm
                    // Foot 关节 ≈ 脚踝，距脚底约 10-12cm
                    // 总缺失高度约 20-24cm
                    // 实测偏小就加大此值，偏大就减小
                    const float HEIGHT_COMPENSATION_MM = 250.0f;  // 默认 22cm 补偿

                    float height_mm = 0.0f;
                    float height_cm = 0.0f;

                    bool head_ok   = P_HEAD.confidence_level   >= K4ABT_JOINT_CONFIDENCE_LOW;
                    bool foot_l_ok = skeleton.joints[K4ABT_JOINT_FOOT_LEFT].confidence_level  >= K4ABT_JOINT_CONFIDENCE_LOW;
                    bool foot_r_ok = skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW;
                    bool ankle_l_ok = P_ANKLE_L.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW;
                    bool ankle_r_ok = P_ANKLE_R.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW;

                    // 方案A：Head → Foot 的 Y 轴垂直距离 + 补偿（最优）
                    if (head_ok && (foot_l_ok || foot_r_ok)) {
                        float foot_y_avg;
                        if (foot_l_ok && foot_r_ok) {
                            foot_y_avg = (skeleton.joints[K4ABT_JOINT_FOOT_LEFT].position.v[1]
                                       +  skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position.v[1]) / 2.0f;
                        } else if (foot_l_ok) {
                            foot_y_avg = skeleton.joints[K4ABT_JOINT_FOOT_LEFT].position.v[1];
                        } else {
                            foot_y_avg = skeleton.joints[K4ABT_JOINT_FOOT_RIGHT].position.v[1];
                        }
                        // Y 轴：下为+，上为−，所以 foot_y − head_y = 正数
                        float vertical_span = foot_y_avg - P_HEAD.position.v[1];
                        height_mm = vertical_span + HEIGHT_COMPENSATION_MM;
                        height_cm = height_mm / 10.0f;
                        printf("Height method: Head->Foot Y-span + %.0fmm = %.1f cm\n", HEIGHT_COMPENSATION_MM, height_cm);
                    }
                    // 方案B：Head → Ankle 的 Y 轴垂直距离 + 更大补偿
                    else if (head_ok && (ankle_l_ok || ankle_r_ok)) {
                        float ankle_y_avg;
                        if (ankle_l_ok && ankle_r_ok) {
                            ankle_y_avg = (P_ANKLE_L.position.v[1] + P_ANKLE_R.position.v[1]) / 2.0f;
                        } else if (ankle_l_ok) {
                            ankle_y_avg = P_ANKLE_L.position.v[1];
                        } else {
                            ankle_y_avg = P_ANKLE_R.position.v[1];
                        }
                        float vertical_span = ankle_y_avg - P_HEAD.position.v[1];
                        height_mm = vertical_span + HEIGHT_COMPENSATION_MM + 70.0f;  // 脚踝到脚底额外约7cm
                        height_cm = height_mm / 10.0f;
                        printf("Height method: Head->Ankle Y-span + %.0fmm = %.1f cm\n", HEIGHT_COMPENSATION_MM + 70.0f, height_cm);
                    }
                    // 方案C：Nose → Ankle 的 Y 轴垂直距离 + 更大补偿（兜底）
                    else {
                        bool nose_ok = P_NOSE.confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW;
                        if (nose_ok && (ankle_l_ok || ankle_r_ok)) {
                            float ankle_y_avg;
                            if (ankle_l_ok && ankle_r_ok) {
                                ankle_y_avg = (P_ANKLE_L.position.v[1] + P_ANKLE_R.position.v[1]) / 2.0f;
                            } else if (ankle_l_ok) {
                                ankle_y_avg = P_ANKLE_L.position.v[1];
                            } else {
                                ankle_y_avg = P_ANKLE_R.position.v[1];
                            }
                            float vertical_span = ankle_y_avg - P_NOSE.position.v[1];
                            height_mm = vertical_span + HEIGHT_COMPENSATION_MM + 180.0f;  // 鼻子以上+脚踝以下约18cm
                            height_cm = height_mm / 10.0f;
                            printf("Height method: Nose->Ankle Y-span + %.0fmm = %.1f cm\n", HEIGHT_COMPENSATION_MM + 180.0f, height_cm);
                        } else {
                            // 最终兜底：旧比例
                            UPPER_BODY = sqrt(
                                pow(P_NECK.position.v[0] - P_PELVIS.position.v[0], 2) +
                                pow(P_NECK.position.v[1] - P_PELVIS.position.v[1], 2) +
                                pow(P_NECK.position.v[2] - P_PELVIS.position.v[2], 2)
                            );
                            height_mm = UPPER_BODY * (1770.0f / 518.0f);
                            height_cm = height_mm / 10.0f;
                            printf("Height method: Neck->Pelvis ratio (last resort) = %.1f cm\n", height_cm);
                        }
                    }

                    // --------------------------
                    // 3. 分开判断：手部动作 和 躯干动作，互不干扰！
                    // --------------------------
                    bool left_hand_over_head = P_WRIST_L.position.v[1] < P_NECK.position.v[1];
                    bool right_hand_over_head = P_WRIST_R.position.v[1] < P_NECK.position.v[1];
                    bool left_hand_reach = P_WRIST_L.position.v[2] < P_SHOULDER_L.position.v[2] - 100;
                    bool right_hand_reach = P_WRIST_R.position.v[2] < P_SHOULDER_R.position.v[2] - 100;

                    string current_hand_action = "Idle";
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

                    // 躯干动作：用膝盖角度
                    string current_body_action = "Standing";
                    float left_knee_angle = calculate_angle(P_HIP_L.position, P_KNEE_L.position, P_ANKLE_L.position);
                    float right_knee_angle = calculate_angle(P_HIP_R.position, P_KNEE_R.position, P_ANKLE_R.position);
                    float avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0f;

                    printf("Knee Angle: Left=%.1f, Right=%.1f, Avg=%.1f\n", left_knee_angle, right_knee_angle, avg_knee_angle);

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
                    // 5. 头部左边画红色文字
                    // --------------------------
                    cv::Point nose_p = joint_to_pixel(P_NOSE, sensor_calibration);
                    if (nose_p.x > 0 && nose_p.y > 0) {
                        char text[128];
                        sprintf(text, "#%d %s | %s | %.1f cm",
                                persistent_id,
                                final_body_action.c_str(),
                                final_hand_action.c_str(),
                                height_cm);
                        cv::putText(
                            cv_rgbImage_no_alpha,
                            text,
                            cv::Point(nose_p.x - 200, nose_p.y),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.8,
                            cv::Scalar(0, 0, 255),
                            2
                        );
                    }

                    // --------------------------
                    // 收集面板数据
                    // --------------------------
                    panel_body_ids.push_back(persistent_id);
                    panel_heights.push_back(height_cm);
                    panel_knee_angles.push_back(avg_knee_angle);
                    panel_body_actions.push_back(final_body_action);
                    panel_hand_actions.push_back(final_hand_action);

                    // --------------------------
                    // 终端输出
                    // --------------------------
                    printf("Tracked #%d | Height: %.1f cm | Body: %s | Hand: %s\n",
                           persistent_id, height_cm,
                           final_body_action.c_str(),
                           final_hand_action.c_str());
                    printf("----------------------------------------\n");
                }
                k4abt_frame_release(body_frame);
            }

            // ---------- 绘制数据可视化侧边栏 ----------
            cv::Mat sidebar = create_sidebar(cv_rgbImage_no_alpha.rows, frame_count, fps,
                                             panel_body_ids.size(),
                                             panel_body_ids, panel_heights, panel_knee_angles,
                                             panel_body_actions, panel_hand_actions);
            // 水平拼接：视频（左） + 侧边栏（右）
            cv::Mat combined;
            cv::hconcat(cv_rgbImage_no_alpha, sidebar, combined);
            cv::imshow("Color (Skeleton)", combined);
        } else {
            // 没有获取到 capture 时也显示拼接画面
            cv::Mat sidebar = create_sidebar(cv_rgbImage_no_alpha.rows, frame_count, fps,
                                             0, {}, {}, {}, {}, {});
            cv::Mat combined;
            cv::hconcat(cv_rgbImage_no_alpha, sidebar, combined);
            cv::imshow("Color (Skeleton)", combined);
        }

        // 深度图窗口保持不变
        cv::imshow("Depth", cv_depth_8U);

        k4a_image_release(rgbImage);
        k4a_image_release(depthImage);

        if (cv::waitKey(1) == 27) {
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