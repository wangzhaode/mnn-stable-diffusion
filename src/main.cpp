#include <iostream>
#include "pipeline.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./main <senetence> <img_name>");
    }
    auto sentence = argv[1];
    auto img_name = argv[2];
    printf("input setnetce: %s\n", sentence);
    printf("output img_name: %s\n", img_name);
    diffusion::Pipeline pipeline("../resource");
    pipeline.run(sentence, img_name);
    return 0;
}