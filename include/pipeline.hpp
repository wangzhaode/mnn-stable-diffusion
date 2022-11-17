#include <map>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

using namespace MNN;
using namespace MNN::Express;

namespace diffusion {

class Pipeline {
public:
    Pipeline(std::string modelPath);
    ~Pipeline() = default;
    bool run(const std::string& sentence, const std::string& img_name);
private:
    void loadNet(std::string modelPath);
    void runNet();
    VARP step_plms(VARP sample, VARP model_output, int index);
    std::unique_ptr<MNN::Tensor> text_encoder(const std::vector<int>& ids);
    VARP unet(std::unique_ptr<MNN::Tensor> text_embeddings);
    VARP vae_decoder(VARP latent);
private:
    std::unique_ptr<MNN::Interpreter> mNet;
    MNN::Session* mSession;
    std::map<std::string, MNN::Tensor*> mInputs, mOutputs;
    std::string mModelPath;
    // step_plms
    std::vector<int> mTimeSteps;
    std::vector<float> mAlphas;
    std::vector<VARP> mEts;
    VARP mSample;
};

}