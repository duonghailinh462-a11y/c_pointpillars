#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float normalize_angle(float a) {
    // normalize to [-pi, pi]
    static constexpr float kPi = 3.14159265358979323846f;
    while (a > kPi) a -= 2 * kPi;
    while (a < -kPi) a += 2 * kPi;
    return a;
}

struct Vec2 { float x, y; };

inline Vec2 rotate(const Vec2& p, float c, float s) {
    return {p.x * c - p.y * s, p.x * s + p.y * c};
}

inline std::array<Vec2, 4> box_corners_bev(const Box3D& b) {
    // center (x,y), size (l along x', w along y') in box local frame
    float hl = b.l * 0.5f;
    float hw = b.w * 0.5f;
    float c = std::cos(b.rot);
    float s = std::sin(b.rot);

    std::array<Vec2, 4> local = {{
        {+hl, +hw},
        {+hl, -hw},
        {-hl, -hw},
        {-hl, +hw},
    }};
    std::array<Vec2, 4> world;
    for (int i = 0; i < 4; ++i) {
        Vec2 r = rotate(local[i], c, s);
        world[i] = {r.x + b.x, r.y + b.y};
    }
    return world;
}

inline float cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }
inline Vec2 sub(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }

bool inside(const Vec2& p, const Vec2& a, const Vec2& b) {
    // left side test for edge a->b
    return cross(sub(b, a), sub(p, a)) >= 0.0f;
}

Vec2 intersection(const Vec2& p1, const Vec2& p2, const Vec2& a, const Vec2& b) {
    // line segment p1->p2 with line a->b (infinite), return intersection.
    Vec2 r = sub(p2, p1);
    Vec2 s = sub(b, a);
    float denom = cross(r, s);
    if (std::fabs(denom) < 1e-8f) {
        return p2; // parallel-ish, fallback
    }
    float t = cross(sub(a, p1), s) / denom;
    return {p1.x + t * r.x, p1.y + t * r.y};
}

float polygon_area(const std::vector<Vec2>& poly) {
    if (poly.size() < 3) return 0.0f;
    float area = 0.0f;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % poly.size()];
        area += p.x * q.y - q.x * p.y;
    }
    return std::fabs(area) * 0.5f;
}

std::vector<Vec2> clip_polygon(const std::vector<Vec2>& subject, const Vec2& a, const Vec2& b) {
    std::vector<Vec2> out;
    if (subject.empty()) return out;
    Vec2 prev = subject.back();
    bool prev_in = inside(prev, a, b);
    for (const auto& cur : subject) {
        bool cur_in = inside(cur, a, b);
        if (cur_in) {
            if (!prev_in) out.push_back(intersection(prev, cur, a, b));
            out.push_back(cur);
        } else if (prev_in) {
            out.push_back(intersection(prev, cur, a, b));
        }
        prev = cur;
        prev_in = cur_in;
    }
    return out;
}

float iou_bev_rotated(const Box3D& a, const Box3D& b) {
    // rotated rectangle IoU in BEV using polygon clipping
    auto ca = box_corners_bev(a);
    auto cb = box_corners_bev(b);

    std::vector<Vec2> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) poly.push_back(ca[i]);

    // clip subject (a) by each edge of b (counter-clockwise)
    for (int i = 0; i < 4; ++i) {
        Vec2 p = cb[i];
        Vec2 q = cb[(i + 1) % 4];
        poly = clip_polygon(poly, p, q);
        if (poly.empty()) break;
    }

    float inter = polygon_area(poly);
    float area_a = a.l * a.w;
    float area_b = b.l * b.w;
    float uni = area_a + area_b - inter;
    if (uni <= 1e-6f) return 0.0f;
    return inter / uni;
}

} // namespace

// -------------------------
// AnchorDecoder
// -------------------------

AnchorDecoder::AnchorDecoder(DecodeConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.grid_x <= 0 || cfg_.grid_y <= 0) {
        throw std::invalid_argument("DecodeConfig: invalid grid size");
    }
    if (cfg_.anchor_sizes.empty()) {
        throw std::invalid_argument("DecodeConfig: anchor_sizes is empty");
    }
    if (cfg_.num_rot <= 0) {
        throw std::invalid_argument("DecodeConfig: num_rot must be > 0");
    }
    if (cfg_.num_classes <= 0) {
        throw std::invalid_argument("DecodeConfig: num_classes must be > 0");
    }
}

std::vector<Box3D> AnchorDecoder::decode(
    const float* box_map,
    const float* score_map,
    float score_thresh) const {
    if (!box_map || !score_map) return {};

    const int H = cfg_.grid_y;
    const int W = cfg_.grid_x;
    const int stride = H * W;

    const int num_types = static_cast<int>(cfg_.anchor_sizes.size());
    const int num_anchors = num_types * cfg_.num_rot;

    // rotations: [0, 1.57] by default
    std::vector<float> rots(cfg_.num_rot, 0.0f);
    if (cfg_.num_rot >= 2) rots[1] = 1.57079632679f;
    for (int i = 2; i < cfg_.num_rot; ++i) rots[i] = rots[i - 1]; // fallback

    std::vector<Box3D> out;
    out.reserve(4096);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pixel = y * W + x;

            // anchor center based on grid cell center
            const float xa = x * cfg_.voxel_size_x + cfg_.x_min + cfg_.voxel_size_x * 0.5f;
            const float ya = y * cfg_.voxel_size_y + cfg_.y_min + cfg_.voxel_size_y * 0.5f;

            for (int a = 0; a < num_anchors; ++a) {
                const int type_idx = a / cfg_.num_rot;
                const int rot_idx = a % cfg_.num_rot;
                const auto& as = cfg_.anchor_sizes[type_idx];

                // score: per-class or single-class
                float best_score = -1e9f;
                int best_cls = 0;
                for (int c = 0; c < cfg_.num_classes; ++c) {
                    const int ch = a * cfg_.num_classes + c;
                    const float logit = score_map[ch * stride + pixel];
                    const float sc = sigmoid(logit);
                    if (sc > best_score) {
                        best_score = sc;
                        best_cls = c;
                    }
                }
                if (best_score < score_thresh) continue;

                // box reg channels: [a*7 + k, y, x]
                const int base_ch = a * 7;
                const float dx = box_map[(base_ch + 0) * stride + pixel];
                const float dy = box_map[(base_ch + 1) * stride + pixel];
                const float dz = box_map[(base_ch + 2) * stride + pixel];
                const float dw = box_map[(base_ch + 3) * stride + pixel];
                const float dl = box_map[(base_ch + 4) * stride + pixel];
                const float dh = box_map[(base_ch + 5) * stride + pixel];
                const float dr = box_map[(base_ch + 6) * stride + pixel];

                const float diagonal = std::sqrt(as.l * as.l + as.w * as.w);

                Box3D b;
                b.x = xa + dx * diagonal;
                b.y = ya + dy * diagonal;
                b.z = as.z_center + dz * as.h;
                b.w = as.w * std::exp(dw);
                b.l = as.l * std::exp(dl);
                b.h = as.h * std::exp(dh);
                b.rot = normalize_angle(rots[rot_idx] + dr);
                b.score = best_score;
                // label：默认用 anchor type（类别）作为 label；如果你的 score 是 per-class，这里更合理用 best_cls
                b.label = (cfg_.num_classes > 1) ? best_cls : type_idx;

                out.push_back(b);
            }
        }
    }

    // 先按 score 排序，减少 NMS 负担
    std::sort(out.begin(), out.end(), [](const Box3D& a, const Box3D& b) { return a.score > b.score; });
    return out;
}

// -------------------------
// NMS (rotated BEV IoU)
// -------------------------

std::vector<Box3D> nms_bev_rotated(const std::vector<Box3D>& boxes, float iou_thr, int max_num) {
    if (boxes.empty()) return {};
    std::vector<int> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);

    // 输入最好已按 score 降序；这里再保险排序一次
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return boxes[a].score > boxes[b].score; });

    std::vector<char> suppressed(boxes.size(), 0);
    std::vector<Box3D> keep;
    keep.reserve(std::min<int>(max_num, static_cast<int>(boxes.size())));

    for (size_t _i = 0; _i < idx.size(); ++_i) {
        int i = idx[_i];
        if (suppressed[i]) continue;

        keep.push_back(boxes[i]);
        if (max_num > 0 && static_cast<int>(keep.size()) >= max_num) break;

        for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
            int j = idx[_j];
            if (suppressed[j]) continue;
            // 常见做法：按类别分别 NMS；这里默认同类才抑制
            if (boxes[j].label != boxes[i].label) continue;

            float iou = iou_bev_rotated(boxes[i], boxes[j]);
            if (iou > iou_thr) suppressed[j] = 1;
        }
    }

    return keep;
}
