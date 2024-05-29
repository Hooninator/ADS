
#ifndef SPGEMM2DMODEL_H
#define SPGEMM2DMODEL_H


#include "common.h"
#include "SpParMatInfo.h"
#include "CommModel.h"
#include "BcastInfo.h"
#include "LocalSpGEMMModel.h"
#include "MergeModel.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"


namespace autotuning {

using namespace combblas;

template <typename AIT, typename ANT,typename ADER,typename BIT,typename BNT,typename BDER>
class SpGEMM2DInputs {

public:

    SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER>()
    {
    }

};

template <typename MT>
class SpGEMM2DModel {
public:

    SpGEMM2DModel(){}

    void Create(PlatformParams& params)
    {
        this->platformParams = params;
        static_cast<MT*>(this)->CreateImpl();
    }


    /* Get runtime estimate of a certain combo of parameters */
    template <typename I>
    std::vector<float> Predict(I& inputs, std::vector<SpGEMMParams>& params) { 
        return static_cast<MT*>(this)->PredictImpl(inputs, params);
    }

    std::vector<float> Predict(std::vector<float>& X) {
        return static_cast<MT*>(this)->PredictImpl(X);
    }

    //TODO: This should be able to return vectors of things other than floats
    template <typename I>
    std::vector<float> MakeFeatureMat(I& inputs, std::vector<SpGEMMParams>& searchSpace) {
        return static_cast<MT*>(this)->MakeFeatureMatImpl(inputs,searchSpace);
    }

    template <typename I>
    std::vector<float> MakeFeatureMat(I& inputs, SpGEMMParams& params) {
        return static_cast<MT*>(this)->MakeFeatureMatImpl(inputs,params);
    }

    //TODO: replace this with somethine non-embarrassing 
#ifdef PROFILE
    void WritePrediction(std::vector<SpGEMMParams>& searchSpace, std::vector<float>& predictions) {
        infoPtr->OFS()<<"----RUNTIME ESTIMATES----"<<std::endl;
        ASSERT(searchSpace.size()==predictions.size(), "sizes not equal");
        for (int i=0; i<searchSpace.size(); i++) {
            infoPtr->OFS()<<searchSpace[i]<<":"<<predictions[i]/1e6<<"s ";
        }
        infoPtr->OFS()<<std::endl;
    }
#endif

protected:

    PlatformParams platformParams;

};


class SpGEMM2DModelAnalytical : public SpGEMM2DModel<SpGEMM2DModelAnalytical> {
public:

    void CreateImpl() {}
    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoAnalytical : public SpParMatInfo<IT,NT,DER> {
    public:
        
        /* (row,col,nnz) */
        //TODO: For col split, no need to store row idx, and for row split, no need to store col idx
        typedef std::vector<std::tuple<IT,IT,IT>> NnzTuples;

        using SpParMatInfo<IT,NT,DER>::SpParMatInfo; 
        using SpParMatInfo<IT,NT,DER>::locNnz; 
        using SpParMatInfo<IT,NT,DER>::locNcolsExact; 
        using SpParMatInfo<IT,NT,DER>::locNrowsExact; 
        using SpParMatInfo<IT,NT,DER>::locNcols; 
        using SpParMatInfo<IT,NT,DER>::locNrows; 
        using SpParMatInfo<IT,NT,DER>::locMat; 
        using SpParMatInfo<IT,NT,DER>::split; 
        using SpParMatInfo<IT,NT,DER>::rowRank; 
        using SpParMatInfo<IT,NT,DER>::colRank; 
        using SpParMatInfo<IT,NT,DER>::gridDims; 
        using SpParMatInfo<IT,NT,DER>::globDensity; 

        SpParMatInfoAnalytical(){}
        
        //TODO: Make this not a pointer
        SpParMatInfoAnalytical(SpParMat<IT,NT,DER> * Mat): 
            SpParMatInfo<IT,NT,DER>(Mat)
        {
            

        }

        /* Create array of tuples containing nnz per tile column for this processor's local tile  */
        NnzTuples * NnzTuplesCol() {

#ifdef PROFILE
            infoPtr->StartTimer("locNnzTuplesColInit");
#endif

            auto _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
            _nnzTuples->reserve(locNcolsExact);

            // Init local data
            int locTupleSize = 0;
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                if (colIter.nnz()>NNZ_THRESH) {
                    _nnzTuples->push_back( std::tuple<IT,IT,IT>{colRank,  colIter.colid() + locNcols*rowRank, colIter.nnz()} );
                }
            }

#ifdef PROFILE
            infoPtr->EndTimer("locNnzTuplesColInit");
#endif

#ifdef DEBUG
            debugPtr->Log("locNnzTuples col");
            for (int i=0; i<_nnzTuples->size(); i++) {
                debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
            }
#endif

            return _nnzTuples;

        }



        /* Initialize array of tuples containing nnz per tile row on this processor's local tile */
        NnzTuples * NnzTuplesRow() {

#ifdef PROFILE
            infoPtr->StartTimer("locNnzTuplesRowInit");
#endif

            // JB: I can't figure out a way to avoid mutating nnz during iteration, so we can't just use std::tuple
            std::map<std::tuple<IT,IT>, IT> nnzMap;
            for (auto colIter = locMat->begcol(); colIter != locMat->endcol(); colIter++) {
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    std::tuple<IT,IT> t{nzIter.rowid() + locNrows*colRank, rowRank};
                    nnzMap.emplace(t, 0);
                    nnzMap[t] += 1;
                }
            }


            auto  _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
            _nnzTuples->reserve(locNrowsExact);

            std::for_each(nnzMap.begin(), nnzMap.end(),
                [&_nnzTuples](auto& elem)  {
                    std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
                    _nnzTuples->push_back( t );
                }
            );

#ifdef PROFILE
            infoPtr->EndTimer("locNnzTuplesRowInit");
#endif

#ifdef DEBUG
            debugPtr->Log("locNnzTuples row");
            for (int i=0; i<_nnzTuples->size(); i++) {
                debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
            }
#endif

            return _nnzTuples;

        }


        /* Approximate local nnz using matrix globDensity
         * This actually just computes the avg nnz per processor
         */
        IT ComputeLocNnzGlobDensity() {

            IT localNcols = std::get<1>(gridDims);
            IT localNrows = std::get<0>(gridDims);
            IT localMatSize = localNcols * localNrows;

            IT localNnzApprox = static_cast<IT>(globDensity * localMatSize);
            return localNnzApprox;
        }


        /* Approximate local nnz using matrix locDensityArr
         */
        IT ComputeLocNnzLocDensity(int procRank) {

            IT localNcols = std::get<1>(gridDims);
            IT localNrows = std::get<0>(gridDims);
            IT localMatSize = localNcols * localNrows;

            IT localNnzApprox = static_cast<IT>(locDensityArr->at(procRank) * localMatSize);
            return localNnzApprox;
        }


        void ComputeNnzArr(SpGEMMParams& params) {

#ifdef PROFILE
            infoPtr->StartTimer("ComputeNnzArr");
#endif

            nnzArr->clear();
            nnzArr->resize(params.GetTotalProcs());

            switch(split) {
                case COL_SPLIT:
                {
                    ComputeNnzArrColSplit(params);
                    break;
                }
                case ROW_SPLIT:
                {
                    ComputeNnzArrRowSplit(params);
                    break;
                }
                default:
                {
                    UNREACH_ERR();
                }
            }

#ifdef PROFILE
            infoPtr->EndTimer("ComputeNnzArr");
#endif

        }

        
        /* Given local nnz in initial 2D processor grid, compute nnz per processor in 3D processr grid
         * WITHOUT explicitly forming the 3D processor grid. */
        void ComputeNnzArrColSplit(SpGEMMParams& params) {

            const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_COL
            // Local nnz array
            std::for_each(nnzTuples->begin(), nnzTuples->end(),
                [&params,this](auto& t) {
                    int i = std::get<0>(t);
                    int j = std::get<1>(t);
                    int owner = ComputeOwnerGrid(params, i*this->locNrows, j, COL_SPLIT);
                    this->nnzArr->at(owner) += std::get<2>(t);
                }
            );
#else
            // Just use local matrix
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                int j = colIter.colid();
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    int i = nzIter.rowid();
                    int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), COL_SPLIT);
                    nnzArr->at(owner) += 1;
                }
            }
#endif

            // Allreduce to get complete counts for each process
            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
         debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr A: "});
#endif

        }
        


        void ComputeNnzArrRowSplit(SpGEMMParams& params) {

            const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_ROW
            // Local data
            std::for_each(nnzTuples->begin(), nnzTuples->end(),
                [&params, this](auto& t) {
                    int i = std::get<0>(t);
                    int j = std::get<1>(t);
                    int owner = ComputeOwnerGrid(params, i, j*this->locNcols, ROW_SPLIT);
                    this->nnzArr->at(owner) += std::get<2>(t);
                }
            );
#else
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                int j = colIter.colid();
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    int i = nzIter.rowid();
                    int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), ROW_SPLIT);
                    nnzArr->at(owner) += 1;
                }
            }
#endif

            // Allreduce to sum all nnz
            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
            debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr B: "});
#endif

        }
        
        IT ComputeLocNnzGrid(NNZ_STRAT strat, int procRank) {
            switch(strat) {
                case NNZ_GLOB_DENSITY:
                    return ComputeLocNnzGlobDensity();
                case NNZ_LOC_DENSITY:
                    return ComputeLocNnzLocDensity(procRank);
                case NNZ_ARR:
                    return nnzArr->at(procRank);
                default:
                    UNREACH_ERR();
            }
            return 0;
        }

    
        int ComputeOwnerGrid(SpGEMMParams& params, const int i, const int j, SPLIT split) {

            const int layers = params.GetLayers();
            const int gridDim = params.GetGridDim();
            const int gridSize = params.GetGridSize();

            IT locNrowsGrid = std::get<0>(gridDims);
            IT locNcolsGrid = std::get<1>(gridDims);

            IT colDiv;
            IT rowDiv;
            IT layerDiv;

            int layerIdx;

            if (split==COL_SPLIT) {
                colDiv = locNcolsGrid*layers;
                rowDiv = locNrowsGrid;
                layerDiv = locNcolsGrid;
                layerIdx = j;
            } else if (split==ROW_SPLIT) {
                colDiv = locNcolsGrid;
                rowDiv = locNrowsGrid*layers;
                layerDiv = locNrowsGrid;
                layerIdx = i;
            }

            const int prow = std::min(static_cast<IT>(i / rowDiv), static_cast<IT>(gridDim-1));
            const int pcol = std::min(static_cast<IT>(j / colDiv), static_cast<IT>(gridDim-1));
            const int player = std::min(static_cast<IT>((layerIdx / layerDiv)%layers), static_cast<IT>(layers-1));

            return (pcol + prow*gridDim + player*gridSize);
        }
    
        
        /* Sum nnz in procRank's row of the hypothetical 3D grid */
        std::vector<IT> SliceNnzRow(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
            return std::vector<IT>(nnzArr->begin()+(procRank/gridDim), nnzArr->begin()+(procRank/gridDim)+gridDim);
        }


        /* Sum nnz in procRank's column of hypothetical 3D grid */
        std::vector<IT> SliceNnzCol(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
            //TODO: Can we use C++17 algorithms for this?
            std::vector<IT> result(gridDim);
            for (int p=0; p<gridDim; p++) {
                result[p] = nnzArr->at((procRank%gridDim)+p*gridDim);
            }
            return result;
        }


        inline std::vector<IT> * GetNnzArr() {return nnzArr;}
        inline std::vector<float> * GetLocDensityArr() const {return locDensityArr;}
        
    private:

        std::vector<float> * locDensityArr;
        NnzTuples * nnzTuples;

        // Stores nnz per processor in hypothetical 3D grid
        std::vector<IT> * nnzArr;
        

    };


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {

    public:
        Inputs(){}

        Inputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpParMat<AIT,ANT,ADER>& A,
                                                    SpParMat<BIT,BNT,BDER>& B):
            Ainfo(&A),Binfo(&B)
        {
        }

        SpParMatInfoAnalytical<AIT,ANT,ADER> Ainfo;
        SpParMatInfoAnalytical<BIT,BNT,BDER> Binfo;
    };


    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> PredictImpl(Inputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, std::vector<SpGEMMParams>& searchSpace) {
        

        std::vector<float> times(searchSpace.size());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("Prediction");
#endif

        std::transform(searchSpace.begin(), searchSpace.end(), times.begin(),
            [&inputs, this](auto& params) {

				auto bcastTime = this->BcastTime<AIT, ANT>(inputs, params); 
                auto localSpGEMMTime = this->LocalSpGEMMTime(inputs, params);
                auto mergeTime = this->MergeTime(inputs, params);
#ifdef PROFILE
                infoPtr->Put("Params", params.OutStr());
                infoPtr->Put("PredBcastTime", bcastTime);
                infoPtr->Put("PredLocalSpGEMMTime", localSpGEMMTime);
                infoPtr->Put("PredMergeTime", mergeTime);
#endif
                return bcastTime + localSpGEMMTime + mergeTime;
            }
        );

#ifdef PROFILE
        infoPtr->WriteInfo();
        infoPtr->EndTimerGlobal("Prediction");
#endif
        return times;

    }


    /* BROADCAST */

    //TODO: Consider nnz estimator class + template to make switching between things here easier
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    float BcastTime(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, SpGEMMParams& params) {

		auto Ainfo = inputs.Ainfo;
		auto Binfo = inputs.Binfo;
		
        auto TreeBcast = [this](int commSize, AIT msgSize) {
            float alpha = this->platformParams.GetInternodeAlpha() * std::log2(commSize);
            float beta = (std::log2(commSize) * msgSize) / this->platformParams.GetInternodeBeta();
            return (alpha + beta) / (1e6);
        };

        auto MsgSize = [](AIT nnz) {
            return nnz*sizeof(ANT) + nnz*sizeof(AIT) + (nnz + 1) * sizeof(AIT);
        };

        float c = (float)(Ainfo.GetNnz()) / (float)(Ainfo.GetNcols());

        AIT nnzA = (c*Ainfo.GetNcols()) / params.GetTotalProcs();
        BIT nnzB = (c*Binfo.GetNcols()) / params.GetTotalProcs();
		
		AIT bytesA = MsgSize(nnzA);
		BIT bytesB = MsgSize(nnzB);
		
		double bcastA = TreeBcast(params.GetGridDim(), bytesA);
		double bcastB = TreeBcast(params.GetGridDim(), bytesB);
		
		return (bcastA + bcastB) * params.GetGridDim();

    }


    /* Local SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    float LocalSpGEMMTime(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, SpGEMMParams& params) {
		auto Ainfo = inputs.Ainfo;
		auto Binfo = inputs.Binfo;
		
		auto FLOPS = [](float c, AIT n, int p){
            float singleMultTime = 2.0*(std::min(1.0, (c/(std::sqrt(p))))) +
                                        ((std::pow(c,2.0)*n) / (std::sqrt(p)*p)) *
                                        std::log2(std::min(n/std::sqrt(p), (std::pow(c,2.0)*n)/(std::sqrt(p)*p)));
            return singleMultTime * std::sqrt(p);
		};

        return FLOPS(Ainfo.GetGlobDensity()*Ainfo.GetNcols(), Ainfo.GetNcols(), params.GetTotalProcs())*
                        this->platformParams.GetCostFLOP();

    }

    /* Local Merge */

    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    float MergeTime(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, SpGEMMParams& params) {

		auto Ainfo = inputs.Ainfo;
		auto Binfo = inputs.Binfo;

        auto FLOPS = [](float c, AIT n, int p) {
            return (std::pow(c,2.0)*n*std::log2(std::sqrt(p))) / p;
        };

        return FLOPS(Ainfo.GetGlobDensity()*Ainfo.GetNcols(), Ainfo.GetNcols(), 
                    params.GetTotalProcs())*
                    this->platformParams.GetCostFLOP();
    }
 
};


/* Precisely compute nnz per process using symbolic grid, and use that for all phases of analytical model */
template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM2DModelAnalyticalPrecise : public SpGEMM2DModel<SpGEMM2DModelAnalyticalPrecise<AIT,ANT,ADER,BIT,BNT,BDER>> {
    
public:
    void CreateImpl(){}

    template <typename IT, typename NT, typename DER>
    class SpParMatInfoAnalyticalPrecise : public SpParMatInfo<IT,NT,DER> {
    public:

        using SpParMatInfo<IT, NT, DER>::locNnz;
        using SpParMatInfo<IT, NT, DER>::locNcolsExact;
        using SpParMatInfo<IT, NT, DER>::locNrowsExact;

        SpParMatInfoAnalyticalPrecise() {}

        SpParMatInfoAnalyticalPrecise(SpParMat<IT, NT, DER> * Mat) {
            gridComm = Mat->getcommgrid()->GetWorld();
            worldSize = Mat->getcommgrid()->GetSize();
        }


        inline MPI_Comm GetGridComm() const { return gridComm; }
        inline int GetWorldSize() const { return worldSize; }

    private:
        MPI_Comm gridComm;
        int worldSize;

    };


    SpGEMM2DModelAnalyticalPrecise() {

		std::vector<std::string> features{
			"nnz-A",
			"nnz-B"
	 	};
		
		nFeatures = features.size();
		
	}


    class Inputs : public SpGEMM2DInputs<AIT, ANT, ADER, BIT, BNT, BDER> {
    public:
        Inputs(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B):
            Ainfo(&A), Binfo(&B)
        {
            ComputeActualDistInfo();
             
        }
        
        SpParMatInfoAnalyticalPrecise<AIT, ANT, ADER> Ainfo;
        SpParMatInfoAnalyticalPrecise<BIT, BNT, BDER> Binfo;
        
        /* Store nnz per processor  */
        struct LocInfo {
            LocInfo(AIT locNnzA, AIT locNnzB):locNnzA(locNnzA),locNnzB(locNnzB){}
            AIT locNnzA;
            AIT locNnzB; //Assume they're the same type to make the MPI call easier
        };
        
        
        void ComputeActualDistInfo() {

            actualDistInfo.reserve(Ainfo.GetWorldSize());

            std::vector<AIT> sendBuf {Ainfo.GetLocNnz(), Binfo.GetLocNnz()};
            AIT * recvBuf = new AIT[sendBuf.size() * Ainfo.GetWorldSize()];
            MPI_Allgather((void*)(sendBuf.data()), sendBuf.size(), MPIType<AIT>(),
                            (void *)(recvBuf), sendBuf.size(), MPIType<AIT>(), 
                            Ainfo.GetGridComm());

            for (int i=0; i<Ainfo.GetWorldSize(); i+=2) {
                actualDistInfo.push_back(new LocInfo{recvBuf[i], recvBuf[i+1]});
            }

        }

        std::vector<LocInfo *> actualDistInfo;

    };

    typedef typename Inputs::LocInfo LocInfo_t;


    std::vector<float> PredictImpl(Inputs& inputs, std::vector<SpGEMMParams>& searchSpace) {
        
        std::vector<float> times(searchSpace.size());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("Prediction");
#endif

        std::vector<float> allTimes(0);
        
        std::transform(searchSpace.begin(), searchSpace.end(), times.begin(),
            [&inputs, &allTimes, this](auto& params) {

#ifdef DEBUG
                debugPtr->Print(params.OutStr());
#endif
                
                auto featureMat = this->MakeFeatureMatImpl(inputs, params);

				auto bcastTime = this->BcastTime(featureMat, params); 
                auto localSpGEMMTime = this->LocalSpGEMMTime(featureMat, params);
                auto mergeTime = this->MergeTime(featureMat, params);
                
                allTimes.clear();
                allTimes.resize(params.GetTotalProcs());

                std::transform(bcastTime.begin(), bcastTime.end(),
                                localSpGEMMTime.begin(),
                                allTimes.begin(), std::plus<float>());

                std::transform(mergeTime.begin(), mergeTime.end(),
                                allTimes.begin(),
                                allTimes.begin(), std::plus<float>());

                return ReduceMax(allTimes);

            }
        );

#ifdef PROFILE
        infoPtr->EndTimerGlobal("Prediction");
#endif
        return times;

    }


    std::vector<float> BcastTime(const std::vector<LocInfo_t *>& featureMat, SpGEMMParams& params) {
    }


    std::vector<float> LocalSpGEMMTime(const std::vector<LocInfo_t *>& featureMat, SpGEMMParams& params) {
    }


    std::vector<float> MergeTime(const std::vector<LocInfo_t *>& featureMat, SpGEMMParams& params) {
    }

    std::vector<LocInfo_t *> MakeFeatureMatImpl(Inputs& inputs, SpGEMMParams& params) {

#ifdef PROFILE
        infoPtr->StartTimer("FeatureCollection");
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        std::vector<LocInfo_t *> featureMat(nFeatures*params.GetTotalProcs());

        // For now, assume always scaling down
        ASSERT(jobPtr->totalTasks>=params.GetTotalProcs(), "Scaling up is not yet supported");

        int gridDim = params.GetGridDim();
        int superTileDim = RoundedSqrt<int,int>(Ainfo.GetWorldSize()) / gridDim;

        auto SuperTileColor = [&gridDim, &superTileDim](int rowRank, int colRank) {
            return ( (rowRank / superTileDim) ) + ( ((colRank) / superTileDim) * gridDim );
        };
        
        auto addLocInfo = [](LocInfo_t * info1, LocInfo_t * info2) {
            return new LocInfo_t(info1->locNnzA + info2->locNnzA, info1->locNnzB + info2->locNnzB);
        };

        for (int k=0; k<Ainfo.GetWorldSize(); k++) {

            int i = k % RoundedSqrt<int,int>(Ainfo.GetWorldSize());
            int j = k / RoundedSqrt<int,int>(Ainfo.GetWorldSize());

            int superTileIdx = SuperTileColor(i, j);
            int startIdx = superTileIdx*nFeatures;
            int endIdx = (superTileIdx+1)*nFeatures;

            std::transform(inputs.actualDistInfo.begin() + (k),
                            inputs.actualDistInfo.begin() + ((k+1)),
                            featureMat.begin() + superTileIdx,
                            featureMat.begin() + superTileIdx,
                            addLocInfo);

        }

#ifdef PROFILE
        infoPtr->EndTimer("FeatureCollection");
#endif
#ifdef DEBUG
        debugPtr->LogVecSameLine(featureMat, "FeatureMat");
#endif

        return featureMat;
        
    }

private:
	int nFeatures;

};

#ifdef XGB_MODEL

class SpGEMM2DModelXgb : public SpGEMM2DModel<SpGEMM2DModelXgb> {
public:

    void CreateImpl() {

        XGB_CHECK(XGBoosterCreate(nullptr, 0, &bstHandle));

        //TODO: Get rid of hardcoded filepath
        const char * modelPath = "../include/CombBLAS/Autotuning/model/model_2d_xgb_globals.model";
        XGB_CHECK(XGBoosterLoadModel(bstHandle, modelPath));
        
        XGB_CHECK(XGBoosterGetNumFeature(bstHandle, (bst_ulong*)(&nFeatures)));
#ifdef DEBUG
        debugPtr->Print0("Num features: " + std::to_string(nFeatures));
#endif
    }

    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoXgb : public SpParMatInfo<IT,NT,DER> {
    public:
        
        using SpParMatInfo<IT,NT,DER>::nnz;
        using SpParMatInfo<IT,NT,DER>::ncols;
        using SpParMatInfo<IT,NT,DER>::nrows;
        using SpParMatInfo<IT,NT,DER>::globDensity;

        SpParMatInfoXgb(SpParMat<IT,NT,DER>& Mat):
            SpParMatInfo<IT,NT,DER>(&Mat)
        {
            
            featureMap.emplace("nnz", nnz);
            featureMap.emplace("m", nrows);
            featureMap.emplace("n", ncols);
            featureMap.emplace("density", globDensity);

            SetGlobalColInfo(Mat);    
        }


        // NOTE: need overloaded function here because behavior differs depending on 2d vs 3d
        void SetGlobalColInfo(SpParMat<IT,NT,DER>& Mat) {
#ifdef PROFILE
            infoPtr->StartTimer("FeatureCollection");
#endif

            // avg nnz per column
            avgNnzCol = static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getncol());

            featureMap.emplace("avgNnzCol", avgNnzCol);

            // avg density per column
            avgDensityCol = (static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getnrow())) / 
                                    static_cast<float>(Mat.getncol());

            featureMap.emplace("avgDensityCol", avgDensityCol);

            // Reduce to get complete nnz per column
            std::vector<IT> nnzColVec(Mat.seqptr()->getncol());
            float sumNnzMeanDiff;

            for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
                nnzColVec[colIter.colid()] = colIter.nnz();
                sumNnzMeanDiff += std::pow( (colIter.nnz() - avgNnzCol), 2);
            }

            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzColVec.data()), nnzColVec.size(), MPIType<IT>(), MPI_SUM,
                        Mat.getcommgrid()->GetColWorld());

            // Compute column densities
            std::vector<float> densityColVec(Mat.seqptr()->getncol());
            float sumDensityMeanDiff;

            std::transform(nnzColVec.begin(), nnzColVec.end(), densityColVec.begin(),
                    [this, &sumDensityMeanDiff](IT nnz) mutable {
                        float d = static_cast<float>(nnz) / static_cast<float>(this->nrows);
                        sumDensityMeanDiff += std::pow( (d - this->avgDensityCol), 2);
                        return d;
                    }
            );

            // Local reduce to get min, max and sum for each column block
            float locMinDensity, locMaxDensity;
            minNnzCol = ReduceMin(nnzColVec);
            maxNnzCol = ReduceMax(nnzColVec);
            minDensityCol = ReduceMin(densityColVec);
            maxDensityCol = ReduceMax(densityColVec);

            // Global reduce to compute final min, max, and sum
            // TODO: use nonblocking collectives?
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&minNnzCol), 1, MPIType<IT>(), MPI_MIN, 
                            Mat.getcommgrid()->GetRowWorld());
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxNnzCol), 1, MPIType<IT>(), MPI_MAX, 
                            Mat.getcommgrid()->GetRowWorld());

            MPI_Allreduce(MPI_IN_PLACE, (void*)(&minDensityCol), 1, MPI_FLOAT, MPI_MIN, 
                            Mat.getcommgrid()->GetRowWorld());
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxDensityCol), 1, MPI_FLOAT, MPI_MAX, 
                            Mat.getcommgrid()->GetRowWorld());

            // pack floats that will be summed into single buffer
            float locBuf[] = {sumNnzMeanDiff, sumDensityMeanDiff};
            MPI_Allreduce(MPI_IN_PLACE, (void*)(locBuf), 2, MPI_FLOAT, MPI_SUM, Mat.getcommgrid()->GetRowWorld());

            // finish stdev calculations
            stdevNnzCol = std::sqrt( sumNnzMeanDiff / Mat.getncol() );
            stdevDensityCol = std::sqrt( sumDensityMeanDiff / Mat.getncol() );
            
            featureMap.emplace("minNnzCol", minNnzCol);
            featureMap.emplace("maxNnzCol", maxNnzCol);
            featureMap.emplace("minDensityCol", minDensityCol);
            featureMap.emplace("maxDensityCol", maxDensityCol);
            featureMap.emplace("stdevNnzCol", stdevNnzCol);
            featureMap.emplace("stdevDensityCol", stdevDensityCol);

#ifdef PROFILE
            infoPtr->EndTimer("FeatureCollection");
            infoPtr->Print("FeatureCollection");
#endif

        }


        inline float GetAvgNnzCol() const {return avgNnzCol;}
        inline IT GetMinNnzCol() const {return minNnzCol;}
        inline IT GetMaxNnzCol() const {return maxNnzCol;}
        inline float GetStdevNnzCol() const {return stdevNnzCol;}

        inline float GetAvgDensityCol() const {return avgDensityCol;}
        inline float GetMinDensityCol() const {return minDensityCol;}
        inline float GetMaxDensityCol() const {return maxDensityCol;}
        inline float GetStdevDensityCol() const {return stdevDensityCol;}

        inline std::map<std::string, float> GetFeatureMap() const {return featureMap;} 

    private:

        float avgNnzCol;
        IT minNnzCol;
        IT maxNnzCol;
        float stdevNnzCol;
        float avgDensityCol;
        float minDensityCol;
        float maxDensityCol;
        float stdevDensityCol;

        std::map<std::string, float> featureMap;

    };

    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {
    public:
        Inputs(SpParMat<AIT,ANT,ADER>& A, SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A),Binfo(B)
        {
        }

        SpParMatInfoXgb<AIT,ANT,ADER> Ainfo;
        SpParMatInfoXgb<BIT,BNT,BDER> Binfo;
        
    };

    std::vector<float> PredictImpl(std::vector<float>& X) {

        // Create DMat
        int nSamples = X.size() / nFeatures;
        DMatrixHandle dMatHandle;
        XGB_CHECK(XGDMatrixCreateFromMat(X.data(), nSamples, nFeatures, 0.0, &dMatHandle)); 

        // Make prediction
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";
        bst_ulong outDim;
        const bst_ulong * outShape; 
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(bstHandle, dMatHandle, config, &outShape, &outDim, &prediction));

        return std::vector<float>(prediction, prediction+nSamples);

    }


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> MakeFeatureMatImpl(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, 
                                            std::vector<SpGEMMParams>& searchSpace) {

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;
        
        int nSamples = searchSpace.size();

        // Feature order
        std::vector<std::string> featureOrder{
            "avgDensityCol",
            "avgNnzCol",
            "density",
            "m",
            "maxDensityCol",
            "maxNnzCol",
            "minDensityCol",
            "minNnzCol",
            "n",
            "nnz",
            "stdevDensityCol",
            "stdevNnzCol"
        };

        // Each row is a sample
        std::vector<float> featureMat;
        featureMat.reserve(nSamples*nFeatures);

        //TODO: There has to be a better way to do this
        // Populate the feature matrix
        for (int i=0; i<nSamples; i++) {

            // Nodes and PPN always go first
            auto currParams = searchSpace[i];
            featureMat.push_back(currParams.GetNodes());
            featureMat.push_back(currParams.GetPPN());
            
            // Iterate through features in this sample according to feature order defined earlier
            // and push them onto the matrix
            std::for_each(featureOrder.begin(), featureOrder.end(),
                [&featureMat, &Ainfo, &Binfo](auto& featureName) {
                    // Order is always feature-A, feature-B
                    featureMat.push_back(Ainfo.GetFeatureMap()[featureName]);
                    featureMat.push_back(Binfo.GetFeatureMap()[featureName]);
                }
            );
        }


        return featureMat; 
    }


private:
    int nFeatures;
    BoosterHandle bstHandle;

};


class SpGEMM2DModelPhase : public SpGEMM2DModel<SpGEMM2DModelPhase> {

public:

    void CreateImpl() {
        XGB_CHECK(XGBoosterCreate(nullptr, 0, &multBstHandle));
        XGB_CHECK(XGBoosterCreate(nullptr, 0, &mergeBstHandle));

        //TODO: Remove hardocded filepaths
        const char * multModelPath = "../include/CombBLAS/Autotuning/model/models/xgb-mult-best.model";
        const char * mergeModelPath = "../include/CombBLAS/Autotuning/model/models/xgb-merge-best.model";
        
        XGB_CHECK(XGBoosterLoadModel(multBstHandle, multModelPath));
        XGB_CHECK(XGBoosterLoadModel(mergeBstHandle, mergeModelPath));

        XGB_CHECK(XGBoosterGetNumFeature(multBstHandle, (bst_ulong*)(&nFeatures)));

        std::vector<std::string> features{
            "FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz-intermediate",
            "outputNnz-final",
            "Nodes",
            "PPN",
        };

        ASSERT(nFeatures==features.size(), "Feature size is wrong");
    }
    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoPhase : public SpParMatInfo<IT,NT,DER> {
    public:
        
        using SpParMatInfo<IT,NT,DER>::locNnz;
        using SpParMatInfo<IT,NT,DER>::locNrowsExact;
        using SpParMatInfo<IT,NT,DER>::locNcolsExact;
        using SpParMatInfo<IT,NT,DER>::rank;
        using SpParMatInfo<IT,NT,DER>::colRank;
        using SpParMatInfo<IT,NT,DER>::rowRank;
        using SpParMatInfo<IT,NT,DER>::ncols;
        using SpParMatInfo<IT,NT,DER>::nrows;

        SpParMatInfoPhase(SpParMat<IT,NT,DER>& Mat):
            SpParMatInfo<IT,NT,DER>(&Mat)
        {
            gridComm = Mat.getcommgrid()->GetWorld();
            worldSize = Mat.getcommgrid()->GetSize();
        }

        MPI_Comm gridComm;
        int worldSize;

    };


    //TODO: One day, this will all need a semiring template parameter
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {
    public:
        
        typedef PlusTimesSRing<ANT,BNT> PTTF;

        Inputs(SpParMat<AIT,ANT,ADER>& A, SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A), Binfo(B), FLOPS(0), outputNnzIntermediate(0), outputNnzFinal(0),
            globalFeatures(0)
        {

#ifdef PROFILE
            infoPtr->StartTimerGlobal("FeatureInit");
#endif

            //ComputeProblemStats(A,B,&outputNnzFinal,&outputNnzIntermediate,&FLOPS);
            ComputeProblemStatsOneSided(A,B, &outputNnzFinal, &outputNnzIntermediate,
                                        &FLOPS);

            std::vector<float> sendBuf{(const float)FLOPS, 
                                        (const float)Ainfo.locNrowsExact,
                                        (const float)Binfo.locNrowsExact,
                                        (const float)Ainfo.locNcolsExact,
                                        (const float)Binfo.locNcolsExact,
                                        (const float)Ainfo.locNnz, 
                                        (const float)Binfo.locNnz, 
                                        (const float)outputNnzIntermediate, 
                                        (const float)outputNnzFinal,
                                        0.0, 0.0}; //These last two placeholders are where nodes and ppn will go
#ifdef DEBUG
            debugPtr->LogVecSameLine(sendBuf, "sendBuf");
#endif

            globalFeatures.resize(sendBuf.size()*Ainfo.worldSize);

            // Gather into globalFeatures 
            // We do an allgather here because I think we'll need to distribute the search space at some point
            MPI_Allgather((void*)(sendBuf.data()), sendBuf.size(), MPI_FLOAT, (void*)(globalFeatures.data()),
                        sendBuf.size(), MPI_FLOAT, Ainfo.gridComm);

#ifdef PROFILE
            infoPtr->EndTimerGlobal("FeatureInit");
#endif
        
        }

		
		template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
		void ComputeProblemStats(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, 
									int64_t * nnzC_SUMMA, int64_t * nnzC_local, int64_t * FLOPS_local)
		{
			typedef typename UDERA::LocalIT LIA;
			typedef typename UDERB::LocalIT LIB;
			static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");

			double t0, t1;

			if(A.getncol() != B.getnrow())
			{
				std::ostringstream outs;
				outs << "Can not multiply, dimensions does not match"<< std::endl;
				outs << A.getncol() << " != " << B.getnrow() << std::endl;
				SpParHelper::Print(outs.str());
				MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
				return;
			}

			int stages, dummy;     // last two parameters of ProductGrid are ignored for Synch multiplication
			std::shared_ptr<CommGrid> GridC = ProductGrid((A.getcommgrid()).get(), (B.getcommgrid()).get(), stages, dummy, dummy);

			MPI_Barrier(GridC->GetWorld());

			LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
			LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
			SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes, (A.getcommgrid())->GetRowWorld());
			SpParHelper::GetSetSizes( *(B.seqptr()), BRecvSizes, (B.getcommgrid())->GetColWorld());

			// Remotely fetched matrices are stored as pointers
			UDERA * ARecv;
			UDERB * BRecv;

			int Aself = (A.getcommgrid())->GetRankInProcRow();
			int Bself = (B.getcommgrid())->GetRankInProcCol();

			double bcastTime = 0;
			double flopTime = 0;
			double nnzTime = 0;

			for(int i = 0; i < stages; ++i)
			{
				std::vector<LIA> ess;
				if(i == Aself)
				{
					ARecv = A.seqptr();    // shallow-copy
				}
				else
				{
					ess.resize(UDERA::esscount);
					for(int j=0; j< UDERA::esscount; ++j)
					{
						ess[j] = ARecvSizes[j][i];        // essentials of the ith matrix in this row
					}
					ARecv = new UDERA();                // first, create the object
				}
#ifdef PROFILE
                t0 = MPI_Wtime();
#endif

				SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);    // then, receive its elements
#ifdef PROFILE
                t1 = MPI_Wtime();
                bcastTime += (t1-t0);
#endif
				ess.clear();

				if(i == Bself)
				{
					BRecv = B.seqptr();    // shallow-copy
				}
				else	
			    {
					ess.resize(UDERB::esscount);
					for(int j=0; j< UDERB::esscount; ++j)
					{
						ess[j] = BRecvSizes[j][i];
					}
					BRecv = new UDERB();
				}

#ifdef PROFILE
                t0 = MPI_Wtime();
#endif
				SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);    // then, receive its elements

#ifdef PROFILE
                t1 = MPI_Wtime();
                bcastTime += (t1-t0);
#endif
				if (BRecv->isZero() || ARecv->isZero()) continue;

#ifdef PROFILE
                t0 = MPI_Wtime();
#endif
				LIB nnzC = estimateNNZFast(*ARecv, *BRecv);
#ifdef PROFILE
                t1 = MPI_Wtime();
                nnzTime += (t1-t0);
#endif
				*nnzC_SUMMA = std::max(nnzC, *nnzC_SUMMA);
#ifdef PROFILE
                t0 = MPI_Wtime();
#endif
				*FLOPS_local += estimateFLOPFast(*ARecv, *BRecv);
#ifdef PROFILE
                t1 = MPI_Wtime();
                flopTime += (t1-t0);
#endif

				if (i==Aself && i==Bself) {
					*nnzC_local = nnzC;
				}

				// delete received data
				if(i != Aself)
					delete ARecv;
				if(i != Bself)
					delete BRecv;
			}

			SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
			SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

#ifdef PROFILE
            infoPtr->PutGlobal("FeatureBcastTime", std::to_string(bcastTime));
            infoPtr->PutGlobal("FeatureNnzInit", std::to_string(nnzTime));
            infoPtr->PutGlobal("FeatureFLOPInit", std::to_string(flopTime));
#endif

		}

		
		template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
		void ComputeProblemStatsOneSided(SpParMat<IU,NU1,UDERA> & A, 
                                        SpParMat<IU,NU2,UDERB> & B, 
                                        int64_t * nnzC_SUMMA, 
                                        int64_t * nnzC_local, int64_t * FLOPS_local)
		{
			typedef typename UDERA::LocalIT LIA;
			typedef typename UDERB::LocalIT LIB;
			static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");

			double t0, t1;

			if(A.getncol() != B.getnrow())
			{
				std::ostringstream outs;
				outs << "Can not multiply, dimensions does not match"<< std::endl;
				outs << A.getncol() << " != " << B.getnrow() << std::endl;
				SpParHelper::Print(outs.str());
				MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
				return;
			}

			int stages, dummy;     
			std::shared_ptr<CommGrid> GridC = ProductGrid((A.getcommgrid()).get(), 
                    (B.getcommgrid()).get(), stages, dummy, dummy);

			MPI_Barrier(GridC->GetWorld());

			LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
			LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
			SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes, 
                    (A.getcommgrid())->GetRowWorld());
			SpParHelper::GetSetSizes( *(B.seqptr()), BRecvSizes, 
                    (B.getcommgrid())->GetColWorld());

			// Remotely fetched matrices are stored as pointers
			UDERA * ARecv;
			UDERB * BRecv;

			int Aself = (A.getcommgrid())->GetRankInProcRow();
			int Bself = (B.getcommgrid())->GetRankInProcCol();

            MPI_Group rowGroup;
            MPI_Group colGroup;
            MPI_Comm_group(A.getcommgrid()->GetRowWorld(), &rowGroup);
            MPI_Comm_group(B.getcommgrid()->GetColWorld(), &colGroup);

            /* Create window objects */
            auto arrInfoA = A.seqptr()->GetArrays();
            auto arrInfoB = B.seqptr()->GetArrays();

            std::vector<MPI_Win> arrwinA(arrInfoA.totalsize());
            std::vector<MPI_Win> arrwinB(arrInfoB.totalsize());
                
            assert(arrInfoA.indarrs.size()==arrInfoB.indarrs.size());
            
            int arrIdx = 0;
            for (int i=0; i<arrInfoA.indarrs.size(); i++) {
                MPI_Win_create(arrInfoA.indarrs[i].addr, 
                                arrInfoA.indarrs[i].count*sizeof(IU),
                                sizeof(IU), MPI_INFO_NULL, //TODO: no locks
                                A.getcommgrid()->GetRowWorld(),
                                &(arrwinA[arrIdx]));
                MPI_Win_create(arrInfoB.indarrs[i].addr, 
                                arrInfoB.indarrs[i].count*sizeof(IU),
                                sizeof(IU), MPI_INFO_NULL, //TODO: no locks
                                B.getcommgrid()->GetColWorld(),
                                &(arrwinB[arrIdx]));
                arrIdx++;
            }

            assert(arrInfoA.numarrs.size()==arrInfoB.numarrs.size());

            for (int i=0; i<arrInfoA.numarrs.size(); i++) {
                MPI_Win_create(arrInfoA.numarrs[i].addr,
                                arrInfoA.numarrs[i].count*sizeof(NU1),
                                sizeof(NU1), MPI_INFO_NULL,
                                A.getcommgrid()->GetRowWorld(),
                                &(arrwinA[arrIdx]));
                MPI_Win_create(arrInfoB.numarrs[i].addr,
                                arrInfoB.numarrs[i].count*sizeof(NU2),
                                sizeof(NU2), MPI_INFO_NULL,
                                B.getcommgrid()->GetColWorld(),
                                &(arrwinB[arrIdx]));
                arrIdx++;
            }


			double fetchTime = 0;
			double flopTime = 0;
			double nnzTime = 0;
            
            LIB flopCLocal = 0;
            LIB * colFlopC = estimateFLOP(*(A.seqptr()), *(B.seqptr()), &flopCLocal);

            *nnzC_local = estimateNNZ_HashFast(*(A.seqptr()), *(B.seqptr()), colFlopC);

			for(int i = 0; i < stages; ++i)
			{
#ifdef PROFILE
                t0 = MPI_Wtime();
#endif

                SpParHelper::LockNFetch(ARecv, i, arrwinA, rowGroup, ARecvSizes);
#ifdef PROFILE
                t1 = MPI_Wtime();
                fetchTime += (t1-t0);
#endif


#ifdef PROFILE
                t0 = MPI_Wtime();
#endif
                SpParHelper::LockNFetch(BRecv, i, arrwinB, colGroup, BRecvSizes);

                SpParHelper::UnlockWindows(i, arrwinA);
                SpParHelper::UnlockWindows(i, arrwinB);
#ifdef PROFILE
                t1 = MPI_Wtime();
                fetchTime += (t1-t0);
#endif
				if (BRecv->isZero() || ARecv->isZero()) continue;
                
                LIB nnzC;
                if (i==Aself && i==Bself) {
                    nnzC = *nnzC_local;
                    *FLOPS_local += flopCLocal;
                } else {
#ifdef PROFILE
                    t0 = MPI_Wtime();
#endif
                    LIB flopC = 0;
                    LIB * colFlopC = estimateFLOP(*ARecv, *BRecv, &flopC);
                    *FLOPS_local += flopC;
#ifdef PROFILE
                    t1 = MPI_Wtime();
                    flopTime += (t1-t0);
#endif
#ifdef PROFILE
                    t0 = MPI_Wtime();
#endif
                  //  nnzC = estimateNNZ_HashFast(*ARecv, *BRecv, colFlopC);
#ifdef PROFILE
                    t1 = MPI_Wtime();
                    nnzTime += (t1-t0);
#endif
                }

				//*nnzC_SUMMA = std::max(nnzC, *nnzC_SUMMA);


				// delete received data
				if(i != Aself)
					delete ARecv;
				if(i != Bself)
					delete BRecv;
			}


			SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
			SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

            for (int i=0; i<arrwinA.size(); i++) {
                MPI_Win_free(&arrwinA[i]);
                MPI_Win_free(&arrwinB[i]);
            }

#ifdef PROFILE
            infoPtr->PutGlobal("FeatureFetchTime", std::to_string(fetchTime));
            infoPtr->PutGlobal("FeatureNnzInit", std::to_string(nnzTime));
            infoPtr->PutGlobal("FeatureFLOPInit", std::to_string(flopTime));
#endif

		}
		

        SpParMatInfoPhase<AIT,ANT,ADER> Ainfo;
        SpParMatInfoPhase<BIT,BNT,BDER> Binfo;
        
        AIT FLOPS;
        AIT outputNnzIntermediate;
        AIT outputNnzFinal;

        std::vector<float> globalFeatures;

    };


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> PredictImpl(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs,
                                    std::vector<SpGEMMParams>& searchSpace) {

        std::vector<float> times(searchSpace.size());

#ifdef PROFILE
        infoPtr->StartTimerGlobal("Prediction");
#endif

        std::transform(searchSpace.begin(), searchSpace.end(), times.begin(),
            [&inputs, this](auto& params) {

#ifdef DEBUG
                debugPtr->Print(params.OutStr());
#endif
                auto featureMat = this->MakeFeatureMatImpl(inputs, params);

                DMatrixHandle featureMatHandle;
                XGB_CHECK(XGDMatrixCreateFromMat(featureMat.data(), params.GetTotalProcs(), nFeatures, 0.0,
                                                    &featureMatHandle));
                
                // Each of these do a prediction for the entire grid
                auto bcastTimes = this->BcastTime<AIT, ANT>(featureMat, params); // Don't need Dmat here
                auto localSpGEMMTimes = this->LocalSpGEMMTime(featureMatHandle, params);
                auto mergeTimes = this->MergeTime(featureMatHandle, params);
                
#ifdef PROFILE
                infoPtr->StartTimer("MaxReduction");
#endif
                // Sum all times, then return the max
                std::vector<float> paramTimes(params.GetTotalProcs());
                std::transform(bcastTimes.begin(), bcastTimes.end(), localSpGEMMTimes.begin(),
                                paramTimes.begin(), std::plus<>());
                std::transform(paramTimes.begin(), paramTimes.end(), mergeTimes.begin(), paramTimes.begin(),
                                std::plus<>());
#ifdef PROFILE
                infoPtr->EndTimer("MaxReduction");
#endif

#ifdef PROFILE
                infoPtr->WriteInfo();
                infoPtr->Clear();
#endif

                return ReduceMax(paramTimes);
            }
        );

#ifdef PROFILE
        infoPtr->EndTimerGlobal("Prediction");
#endif
        return times;

    }


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> MakeFeatureMatImpl(Inputs<AIT,ANT,ADER, BIT,BNT,BDER>& inputs,
                                            SpGEMMParams& params) {


#ifdef PROFILE
        infoPtr->StartTimer("FeatureCollection");
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        std::vector<float> featureMat(nFeatures*params.GetTotalProcs());

        // For now, assume always scaling down
        ASSERT(jobPtr->totalTasks>=params.GetTotalProcs(), "Scaling up is not yet supported");

        int gridDim = params.GetGridDim();
        int superTileDim = RoundedSqrt<int,int>(Ainfo.worldSize) / gridDim;

        auto SuperTileColor = [&gridDim, &superTileDim](int rowRank, int colRank) {
            return ( (rowRank / superTileDim) ) + ( ((colRank) / superTileDim) * gridDim );
        };

        // Reduce into featureMat
        for (int k=0; k<Ainfo.worldSize; k++) {

            int i = k % RoundedSqrt<int,int>(Ainfo.worldSize);
            int j = k / RoundedSqrt<int,int>(Ainfo.worldSize);
            
            int superTileIdx = SuperTileColor(i, j);
#ifdef DEBUG
            //debugPtr->Print("Rank " + std::to_string(k) + " mapped to " + std::to_string(superTileIdx));
#endif
            
            int startIdx = superTileIdx*nFeatures;
            int endIdx = (superTileIdx+1)*nFeatures;

            std::transform(inputs.globalFeatures.begin() + (k*nFeatures), 
                            inputs.globalFeatures.begin() + ((k+1)*nFeatures),
                            featureMat.begin() + superTileIdx*nFeatures,
                            featureMat.begin() + superTileIdx*nFeatures,
                            std::plus<>());

        }

        // Reduction for Nodes and PPN is not required
        for (int k=0; k<params.GetTotalProcs(); k++) {
            featureMat[k*nFeatures + (nFeatures-2)] = params.GetNodes();
            featureMat[k*nFeatures + (nFeatures-1)] = params.GetPPN();

            // I should be arrested for writing this
            featureMat[k*nFeatures + 1] = (std::ceil(Ainfo.GetNrows() / gridDim));
            featureMat[k*nFeatures + 2] = (std::ceil(Binfo.GetNrows() / gridDim));
            featureMat[k*nFeatures + 3] = (std::ceil(Ainfo.GetNcols() / gridDim));
            featureMat[k*nFeatures + 4] = (std::ceil(Binfo.GetNcols() / gridDim));
        }


#ifdef PROFILE
        infoPtr->EndTimer("FeatureCollection");
#endif
#ifdef DEBUG
        debugPtr->LogVecSameLine(featureMat, "FeatureMat");
#endif


        return featureMat;

    }

    template <typename IT, typename NT>
    std::vector<float> BcastTime(std::vector<float>& X, SpGEMMParams& params) {
        
#ifdef PROFILE
        infoPtr->StartTimer("BcastCompute");
#endif

        auto TreeBcast = [this](int commSize, IT msgSize) {
            float alpha = this->platformParams.GetInternodeAlpha() * std::log2(commSize);
            float beta = (std::log2(commSize) * msgSize) / this->platformParams.GetInternodeBeta();
            return (alpha + beta)/(1e6);
        };

        auto MsgSize = [](IT nnz) {
            return nnz*sizeof(NT) + nnz*sizeof(IT) + (nnz + 1) * sizeof(IT);
        };


        // Compute each local bcast time
        std::vector<float> timesA(params.GetGridDim());
        std::vector<float> timesB(params.GetGridDim());
        for (int k=0; k<params.GetTotalProcs(); k++) {

            IT nnzA = static_cast<IT>(X[k*nFeatures + 5]); //TODO: Hardcoding these numbers makes me ill
            IT nnzB = static_cast<IT>(X[k*nFeatures + 6]);

            IT bytesA = MsgSize(nnzA);
            IT bytesB = MsgSize(nnzB);
            
            float bcastTimeA = TreeBcast(params.GetGridDim(), bytesA); 
            float bcastTimeB = TreeBcast(params.GetGridDim(), bytesB); 
            
            int i = k % params.GetGridDim();
            int j = k / params.GetGridDim();

            timesA[i] += bcastTimeA;
            timesB[j] += bcastTimeB;

        }


        // Compute final array of bcast times
        std::vector<float> finalTimes(params.GetTotalProcs());
        for (int k=0; k<finalTimes.size(); k++) {
            
            int i = k % params.GetGridDim();
            int j = k / params.GetGridDim();

            finalTimes[k] = timesA[i] + timesB[j];

        }
        
#ifdef PROFILE
        infoPtr->EndTimer("BcastCompute");
#endif

        return finalTimes;
    }

    
    std::vector<float> LocalSpGEMMTime(DMatrixHandle& X, SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("MultCompute");
#endif

        //TODO: Does this matter?
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";

        bst_ulong outDim;
        const bst_ulong * outShape;
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(multBstHandle, X, config, &outShape, &outDim, &prediction));

#ifdef PROFILE
        infoPtr->EndTimer("MultCompute");
#endif

        return std::vector<float>(prediction, prediction+params.GetTotalProcs());
  
    }


    std::vector<float> MergeTime(DMatrixHandle& X, SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("MergeCompute");
#endif
        //TODO: Does this matter?
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";

        bst_ulong outDim;
        const bst_ulong * outShape;
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(mergeBstHandle, X, config, &outShape, &outDim, &prediction));

#ifdef PROFILE
        infoPtr->EndTimer("MergeCompute");
#endif

        return std::vector<float>(prediction, prediction+params.GetTotalProcs());
    }


private:
    int nFeatures; // same number of features for both models
    BoosterHandle multBstHandle;
    BoosterHandle mergeBstHandle;

};

#endif




}//autotuning

#endif





