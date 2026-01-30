#pragma once

#include <array>
#include <cstdint>
#include <vector>

class PostProcessor {
public:
    PostProcessor(float score_thr = 0.3f, float nms_thr = 0.01f, int max_num = 100);

    // 旧版：给 end2end_sim.onnx 的 bboxes/scores 做简单筛选（目前工程在用）
    DetectionResult process(
        const std::vector<float>& bboxes,
        const std::vector<float>& scores,
        const std::vector<int64_t>& bbox_shape,
        const std::vector<int64_t>& score_shape);

private:
    float score_thr_;
    float nms_thr_;
    int max_num_;

    float compute_iou_3d(const std::array<float, 7>& box1, const std::array<float, 7>& box2);
};

// =========================
// 新版：RPN 裸 head（NCHW）decode + NMS
// =========================
struct Box3D {
    float x = 0, y = 0, z = 0;
    float w = 0, l = 0, h = 0;
    float rot = 0;
    float score = 0;
    int label = 0;
};

struct DecodeConfig {
    // BEV 网格
    int grid_x = 432;
    int grid_y = 496;
    float voxel_size_x = 0.16f;
    float voxel_size_y = 0.16f;
    float x_min = 0.0f;
    float y_min = -39.68f;

    // anchors: 每个类别一个 size，且每个位置有 num_rot 个角度
    struct AnchorSize { float w, l, h, z_center; };
    std::vector<AnchorSize> anchor_sizes = {
        {1.6f, 3.9f, 1.56f, -1.78f},  // Car (Index 0)
        {0.6f, 0.8f, 1.73f, -0.6f},   // Pedestrian (Index 1)
        {0.6f, 1.76f, 1.73f, -0.6f},  // Cyclist (Index 2)
    };
    int num_rot = 2; // 0, 1.57

    // head 形状解释：
    // box_map:  [1, num_anchors*7, H, W]，按 NCHW
    // score_map:[1, num_anchors*num_classes, H, W]（per-class）
    // 若你的 score 是 per-anchor 单通道，把 num_classes 设为 1 即可。
    int num_classes = 3;
};

class AnchorDecoder {
public:
    explicit AnchorDecoder(DecodeConfig cfg);
    const DecodeConfig& cfg() const { return cfg_; }

    // 输入为 NCHW（float32）裸输出指针
    std::vector<Box3D> decode(const float* box_map, const float* score_map, float score_thresh) const;

private:
    DecodeConfig cfg_;
};

std::vector<Box3D> nms_bev_rotated(
    const std::vector<Box3D>& boxes,
    float iou_thr,
    int max_num);
