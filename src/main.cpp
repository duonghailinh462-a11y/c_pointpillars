#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "postprocess.h"

static std::vector<float> load_f32_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("无法打开文件: " + path);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("文件大小不是float32的整数倍: " + path);
    }
    std::vector<float> data(static_cast<size_t>(size / sizeof(float)));
    if (!f.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("读取失败: " + path);
    }
    return data;
}

static void print_boxes(const std::vector<Box3D>& boxes) {
    std::cout << "\n========================================\n";
    std::cout << "Decode + NMS Results\n";
    std::cout << "========================================\n";
    std::cout << "num: " << boxes.size() << "\n";
    int n = std::min<int>(10, static_cast<int>(boxes.size()));
    for (int i = 0; i < n; ++i) {
        const auto& b = boxes[i];
        printf("%2d | score=%.4f label=%d | [x=%.2f y=%.2f z=%.2f w=%.2f l=%.2f h=%.2f rot=%.2f]\n",
               i, b.score, b.label, b.x, b.y, b.z, b.w, b.l, b.h, b.rot);
    }
}

int main(int argc, char** argv) {
    std::string box_map_path;
    std::string score_map_path;
    float score_thr = 0.3f;
    float nms_thr = 0.01f;
    int max_num = 100;

    DecodeConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--box-map" && i + 1 < argc) {
            box_map_path = argv[++i];
        } else if (arg == "--score-map" && i + 1 < argc) {
            score_map_path = argv[++i];
        } else if (arg == "--score-thr" && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        } else if (arg == "--nms-thr" && i + 1 < argc) {
            nms_thr = std::stof(argv[++i]);
        } else if (arg == "--max-num" && i + 1 < argc) {
            max_num = std::stoi(argv[++i]);
        } else if (arg == "--num-classes" && i + 1 < argc) {
            cfg.num_classes = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "用法: " << argv[0] << " --box-map box.bin --score-map score.bin [options]\n"
                << "options:\n"
                << "  --score-thr <float>      (default 0.3)\n"
                << "  --nms-thr <float>        (default 0.01)\n"
                << "  --max-num <int>          (default 100)\n"
                << "  --num-classes <int>      score head 的类别数（per-class），若是单通道设为1\n";
            return 0;
        }
    }

    if (box_map_path.empty() || score_map_path.empty()) {
        std::cerr << "缺少参数：必须提供 --box-map 和 --score-map（float32 NCHW 的裸输出）\n";
        return 1;
    }

    // 读取裸输出
    auto box_map = load_f32_bin(box_map_path);
    auto score_map = load_f32_bin(score_map_path);

    // decode + nms
    AnchorDecoder decoder(cfg);
    auto decoded = decoder.decode(box_map.data(), score_map.data(), score_thr);
    auto final_boxes = nms_bev_rotated(decoded, nms_thr, max_num);

    print_boxes(final_boxes);
    return 0;
}
