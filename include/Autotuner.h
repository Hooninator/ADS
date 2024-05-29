
#ifndef AUTOTUNER_H
#define AUTOTUNER_H


#include "common.h"
#include "SpGEMM2DModel.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"

namespace autotuning {


using namespace combblas;

class Autotuner {

public:
    
    
    /* CONSTRUCTORS */
    

    // Assumes PlatformParams has already been constructed
    Autotuner(PlatformParams& params): platformParams(params) {
        ASSERT(initCalled, "Please call autotuning::Init() first.");
    }
    
    
    /* TUNING */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DAnalytical(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                    std::string& matpathA, std::string& matpathB,
                                    uint32_t maxNodes = 0, uint32_t maxPPN = 0)
    {

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+"-"+
                                std::to_string(autotuning::rank)+".out", 
                                autotuning::rank);
#endif

        MPI_Barrier(A.getcommgrid()->GetWorld());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DAnalytical");
#endif

        typedef SpGEMM2DModel<SpGEMM2DModelAnalytical> ModelType;
        ModelType model;
        model.Create(platformParams);
        
#ifdef PROFILE
        infoPtr->StartTimerGlobal("Inputs");
#endif
        SpGEMM2DModelAnalytical::Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A, B);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("Inputs");
        infoPtr->PrintGlobal("Inputs");
#endif
        
        if (maxNodes==0)
            maxNodes = jobPtr->nodes; //if maxNodes not specified, assume we can scale to max number of nodes in job
        if (maxPPN==0)
            maxPPN = jobPtr->tasksPerNode;
                                    
        SpGEMMParams resultParams; 
        std::vector<SpGEMMParams> searchSpace = SpGEMMParams::ConstructSearchSpace2D(platformParams, maxNodes, maxPPN);
        resultParams = SearchBruteForce<SpGEMMParams, ModelType>(inputs, model, searchSpace);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DAnalytical");
        infoPtr->PrintGlobal("TuneSpGEMM2DAnalytical");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        MPI_Barrier(A.getcommgrid()->GetWorld());

        return resultParams;

    }

    //TODO: This should really be completely changed so the model type is a template parameter...
    // this is literally just the same as the method above this, but with a different model type...
    // will also need some way to match model types to tuning methods, since not every model supports all methods
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DAnalyticalPrecise(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                        std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+"-"+std::to_string(autotuning::rank)+".out", autotuning::rank);
#endif

        MPI_Barrier(A.getcommgrid()->GetWorld());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DAnalyticalPrecise");
#endif
        
        typedef SpGEMM2DModelAnalyticalPrecise<AIT,ANT,ADER,BIT,BNT,BDER> ModelDerType;
        typedef SpGEMM2DModel<ModelDerType> ModelType;
        ModelType model;
        model.Create(platformParams);
        
#ifdef PROFILE
        infoPtr->StartTimerGlobal("Inputs");
#endif
        typename ModelDerType::Inputs inputs(A, B);
#ifdef PROFILE
        infoPtr->EndTimerGlobal("Inputs");
        infoPtr->PrintGlobal("Inputs");
#endif
        
        SpGEMMParams resultParams; 
        std::vector<SpGEMMParams> searchSpace = SpGEMMParams::ConstructSearchSpace2D(platformParams, jobPtr->nodes, jobPtr->tasksPerNode);
        resultParams = SearchBruteForce<SpGEMMParams, ModelType>(inputs, model, searchSpace);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DAnalyticalPrecise");
        infoPtr->PrintGlobal("TuneSpGEMM2DAnalyticalPrecise");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }


#ifdef XGB
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DXgb(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                    std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+".out", autotuning::rank);
#endif

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DXgb");
#endif

        typedef SpGEMM2DModel<SpGEMM2DModelXgb> ModelType;
        ModelType model;
        model.Create(platformParams);
        
        SpGEMM2DModelXgb::Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A, B);
        
        SpGEMMParams resultParams; 
        resultParams = SearchInference<SpGEMMParams>(inputs, model);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DXgb");
        infoPtr->PrintGlobal("TuneSpGEMM2DXgb");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }
#endif


#ifdef XGB

    //TODO: This should probably just be a tuneinference function with a template parameter
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DPhase(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                    std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+"-"+std::to_string(autotuning::rank)+".out", autotuning::rank);
#endif

        MPI_Barrier(A.getcommgrid()->GetWorld());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DPhase");
#endif

        typedef SpGEMM2DModel<SpGEMM2DModelPhase> ModelType;
        ModelType model;
        model.Create(platformParams);
        
        SpGEMM2DModelPhase::Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A, B);
        
        SpGEMMParams resultParams; 
        std::vector<SpGEMMParams> searchSpace = SpGEMMParams::ConstructSearchSpace2D(platformParams, jobPtr->nodes, jobPtr->tasksPerNode);
        resultParams = ParSearchBruteForce<SpGEMMParams, ModelType>(inputs, model, searchSpace);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DPhase");
        infoPtr->PrintGlobal("TuneSpGEMM2DPhase");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }
#endif

    template <typename P, typename M, typename I>
    P SearchBruteForce(I& inputs, M& model, std::vector<P>& searchSpace) {

#ifdef PROFILE
        infoPtr->StartTimerGlobal("BruteForceSearch");
#endif

        ASSERT(searchSpace.size()>0, "Global search space is of size 0!");

#ifdef PROFILE
        infoPtr->PutGlobal("SearchSpaceSize", std::to_string(searchSpace.size()));
#endif

        std::vector<float> predictions = model.Predict(inputs, searchSpace);

#ifdef DEBUG
        debugPtr->Print("Searching for min");
#endif

        P bestParams;  
        bestParams = searchSpace[std::distance(predictions.begin(), std::min_element(predictions.begin(), predictions.end()))];

#ifdef PROFILE

        model.WritePrediction(searchSpace, predictions);

        infoPtr->PutGlobal("BestParams", bestParams.OutStr());
        infoPtr->PutGlobal("PredSpGEMMTime", std::to_string(ReduceMin(predictions)));

        infoPtr->EndTimerGlobal("BruteForceSearch");
        infoPtr->PrintGlobal("BruteForceSearch");
        infoPtr->PrintGlobal("BestParams");
        infoPtr->PrintGlobal("PredSpGEMMTime");
#endif
        
        return bestParams;
    }
    
    template <typename P, typename M, typename I>
    P ParSearchBruteForce(I& inputs, M& model, std::vector<P>& searchSpace) {

#ifdef PROFILE
        infoPtr->StartTimerGlobal("BruteForceSearch");
#endif

        std::vector<P> localSpace;
        std::vector<int> recvCounts(autotuning::worldSize);
        int partitionSize = std::max(1, (int)(searchSpace.size() / autotuning::worldSize));
        for (int i=0; i<searchSpace.size(); i++) {
            int target = std::min(i / partitionSize, autotuning::worldSize);
            if (target==autotuning::rank) {
                localSpace.push_back(searchSpace[i]);
            }
            recvCounts[target] += 1;
        }
        
        std::vector<int> displs(autotuning::worldSize);
        for (int i=1; i<displs.size(); i++) {
            displs[i] = displs[i-1] + recvCounts[i-1];
        }


        ASSERT(searchSpace.size()>0, "Global search space is of size 0!");

#ifdef PROFILE
        infoPtr->PutGlobal("SearchSpaceSize", std::to_string(searchSpace.size()));
        infoPtr->PutGlobal("LocalSearchSpaceSize", std::to_string(localSpace.size()));
#endif

        std::vector<float> localPredictions = model.Predict(inputs, localSpace);
#ifdef DEBUG
        debugPtr->Print("Predictions done, beginning allgather, local size " + std::to_string(localPredictions.size()));
#endif

        std::vector<float> predictions(searchSpace.size());
        MPI_Allgatherv((void*)(localPredictions.data()), localPredictions.size(), MPI_FLOAT,
                        (void*)(predictions.data()), recvCounts.data(), displs.data(),
                        MPI_FLOAT, MPI_COMM_WORLD); //TODO: Make an autotuning::commWorld and use that here


#ifdef DEBUG
        debugPtr->Print("Searching for min");
#endif

        P bestParams;  
        bestParams = searchSpace[std::distance(predictions.begin(), std::min_element(predictions.begin(), predictions.end()))];

#ifdef PROFILE

        model.WritePrediction(searchSpace, predictions);

        infoPtr->PutGlobal("BestParams", bestParams.OutStr());
        infoPtr->PutGlobal("PredSpGEMMTime", std::to_string(ReduceMin(predictions)));

        infoPtr->EndTimerGlobal("BruteForceSearch");
        infoPtr->PrintGlobal("BruteForceSearch");
        infoPtr->PrintGlobal("BestParams");
        infoPtr->PrintGlobal("PredSpGEMMTime");
#endif
        
        return bestParams;
    }


    template <typename P, typename M, typename I>
    P SearchInference(I& inputs, M& model, std::vector<P>& searchSpace) {
        
#ifdef PROFILE
        infoPtr->StartTimerGlobal("InferenceSearch");
#endif

        ASSERT(searchSpace.size()>0, "Search space is of size 0!");

#ifdef PROFILE
        infoPtr->StartTimer("FeatureMat");
#endif

        //NOTE: FeatureMat is in row-major order
        std::vector<float> featureMat;
        featureMat = model.MakeFeatureMat(inputs, searchSpace);

#ifdef PROFILE
        infoPtr->EndTimer("FeatureMat");
        infoPtr->Print("FeatureMat");
#endif

#ifdef PROFILE
        infoPtr->StartTimer("Prediction");
#endif
        
        std::vector<float> predictions;
        predictions = model.Predict(featureMat);

#ifdef PROFILE
        infoPtr->EndTimer("Prediction");
        infoPtr->Print("Prediction");

        model.WritePrediction(searchSpace, predictions);

#endif

        auto minElem = std::min_element(predictions.begin(), predictions.end());
        int minIdx = std::distance(predictions.begin(), minElem);

        P bestParams = searchSpace[minIdx];

#ifdef PROFILE
        infoPtr->PutGlobal("BestParams", bestParams.OutStr());
        infoPtr->PutGlobal("BestTime", std::to_string(predictions[minIdx]));
        infoPtr->EndTimerGlobal("InferenceSearch");
        infoPtr->PrintGlobal("InferenceSearch");
        infoPtr->PrintGlobal("BestParams");
        infoPtr->PrintGlobal("BestTime");
#endif

        return bestParams;

    }

    ~Autotuner(){}

private:
    PlatformParams platformParams;

};//Autotuner


}//combblas
#endif
