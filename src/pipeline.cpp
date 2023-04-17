#include <random>
#include <fstream>
#include <chrono>
#include "pipeline.hpp"
#include "tokenizer.hpp"
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>

using namespace CV;

namespace diffusion {

void display_progress(int cur, int total){
    putchar('\r');
    printf("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    printf("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

Pipeline::Pipeline(std::string modelPath) : mModelPath(modelPath) {
    std::ifstream alphaFile(modelPath + "/alphas.txt");
    int index = 0;
    float alpha;
    while (alphaFile >> alpha) {
        mAlphas.push_back(alpha);
    }
    mTimeSteps = {
        981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1
    };
}

void Pipeline::loadNet(std::string modelPath) {
    mNet.reset(Interpreter::createFromFile(modelPath.c_str()));
    ScheduleConfig config;
#if 1
    config.type = MNN_FORWARD_CUDA;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
#else
    config.type = MNN_FORWARD_CPU;
    config.numThread = 12;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
#endif
    config.backendConfig  = &backendConfig;
    mSession = mNet->createSession(config);
    mNet->releaseModel();
}

void Pipeline::runNet() {
    // AUTOTIME;
    auto t1 =  std::chrono::high_resolution_clock::now();
    mNet->runSession(mSession);
    auto t2 =  std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf(" [iter time: %f ms]", time / 1000.0);
}

std::unique_ptr<Tensor> Pipeline::text_encoder(const std::vector<int>& ids) {
    loadNet(mModelPath + "/text_encoder.mnn");
    auto input = mNet->getSessionInput(mSession, NULL);
    auto output = mNet->getSessionOutput(mSession, NULL);
    std::unique_ptr<Tensor> idsTensor(Tensor::create(input->shape(), input->getType(), const_cast<int*>(ids.data()), input->getDimensionType()));
    input->copyFromHostTensor(idsTensor.get());
    runNet();
    std::unique_ptr<Tensor> text_embeddings(new Tensor(output, output->getDimensionType()));
    output->copyToHostTensor(text_embeddings.get());
    mNet.reset();
    return text_embeddings;
}

VARP Pipeline::step_plms(VARP sample, VARP model_output, int index) {
    int timestep = mTimeSteps[index];
    int prev_timestep = 0;
    if (index + 1 < mTimeSteps.size()) {
        prev_timestep = mTimeSteps[index + 1];
    }
    if (index != 1) {
        if (mEts.size() >= 4) {
            mEts[mEts.size() - 4] = nullptr;
        }
        mEts.push_back(model_output);
    } else {       
        timestep = mTimeSteps[0];
        prev_timestep = mTimeSteps[1];
    }
    int ets = mEts.size() - 1;
    if (index == 0) {
        mSample = sample;
    } else if (index == 1) {
        model_output = (model_output + mEts[ets]) * _Const(0.5);
        sample = mSample;
    } else if (ets == 1) {
        model_output = (_Const(3.0) * mEts[ets] - mEts[ets-1]) * _Const(0.5);
    } else if (ets == 2) {
        model_output = (_Const(23.0) * mEts[ets] - _Const(16.0) * mEts[ets-1] + _Const(5.0) * mEts[ets-2]) * _Const(1.0 / 12.0);
    } else if (ets >= 3) {
        model_output = _Const(1. / 24.) * (_Const(55.0) * mEts[ets] - _Const(59.0) * mEts[ets-1] + _Const(37.0) * mEts[ets-2] - _Const(9.0) * mEts[ets-3]);
    }
    auto alpha_prod_t = mAlphas[timestep];
    auto alpha_prod_t_prev = mAlphas[prev_timestep];
    auto beta_prod_t = 1 - alpha_prod_t;
    auto beta_prod_t_prev = 1 - alpha_prod_t_prev;
    auto sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);
    auto model_output_denom_coeff = alpha_prod_t * std::sqrt(beta_prod_t_prev) + std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);
    auto prev_sample = _Scalar(sample_coeff) * sample - _Scalar((alpha_prod_t_prev - alpha_prod_t)/model_output_denom_coeff) * model_output;
    return prev_sample;
}

VARP Pipeline::unet(std::unique_ptr<Tensor> text_embeddings) {
    loadNet(mModelPath + "/unet.mnn");
    auto sample = mNet->getSessionInput(mSession, "sample");
    auto timestep = mNet->getSessionInput(mSession, "timestep");
    auto encoder_hidden_states = mNet->getSessionInput(mSession, "encoder_hidden_states");
    auto output = mNet->getSessionOutput(mSession, NULL);

    std::unique_ptr<Tensor> latents(new Tensor(sample, Tensor::CAFFE));
    std::unique_ptr<Tensor> timestepVal(new Tensor(timestep, timestep->getDimensionType()));
    std::unique_ptr<Tensor> pred(new Tensor(output, output->getDimensionType()));

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> initVal(16384);
    for (int i = 0; i < 16384; i++) {
        initVal[i] = normal(rng);
    }
    VARP latentvar = _Const(initVal.data(), {1, 4, 64, 64}, NCHW);
    int zero = 0, one = 1;
    for (int i = 0; i < mTimeSteps.size(); i++) {
        display_progress(i, 50);
        memcpy(latents->host<void>(), latentvar->readMap<void>(), 65536);
        memcpy(latents->host<char>() + 65536, latentvar->readMap<void>(), 65536);
        timestepVal->host<float>()[0] = mTimeSteps[i];
        sample->copyFromHostTensor(latents.get());
        timestep->copyFromHostTensor(timestepVal.get());
        encoder_hidden_states->copyFromHostTensor(text_embeddings.get());
        runNet();
        output->copyToHostTensor(pred.get());
        auto noise_pred = Variable::create(Expr::create(pred.get(), false));
        auto noise_pred_uncond = _Gather(noise_pred, _Const(&zero, {1}, NHWC, halide_type_of<int>()));
        auto noise_pred_text = _Gather(noise_pred, _Const(&one, {1}, NHWC, halide_type_of<int>()));
        noise_pred = _Const(7.5) * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond;
        latentvar = step_plms(latentvar, noise_pred, i);
    }
    latentvar.fix(VARP::CONSTANT);
    return latentvar;
}

VARP Pipeline::vae_decoder(VARP latent) {
    latent = latent * _Const(1 / 0.18215);
    loadNet(mModelPath + "/vae_decoder.mnn");
    auto input = mNet->getSessionInput(mSession, NULL);
    auto output = mNet->getSessionOutput(mSession, NULL);
    std::unique_ptr<Tensor> latentTensor(Tensor::create(input->shape(), input->getType(), const_cast<void*>(latent->readMap<void>()), input->getDimensionType()));
    input->copyFromHostTensor(latentTensor.get());
    runNet();
    std::unique_ptr<Tensor> sampleTensor(new Tensor(output, output->getDimensionType()));
    output->copyToHostTensor(sampleTensor.get());
    auto image = Variable::create(Expr::create(sampleTensor.get(), false));
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    image = cvtColor(image, COLOR_BGR2RGB);
    image.fix(VARP::CONSTANT);
    return image;
}

bool Pipeline::run(const std::string& sentence, const std::string& img_name) {
    diffusion::tokenizer tok(mModelPath + "/vocab.txt");
    auto ids = tok.sentence(sentence, 512);
    auto text_embeddings = text_encoder(ids);
    auto latent = unet(std::move(text_embeddings));
    auto image = vae_decoder(latent);
    bool res = imwrite(img_name, image);
    if (res) printf("SUCCESS! write to %s\n", img_name.c_str());
    return res;
}

}
