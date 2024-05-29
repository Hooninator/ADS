
#ifndef SPPARMATINFO_H
#define SPPARMATINFO_H

#include "common.h"
#include "SpGEMMParams.h"


#define NNZ_THRESH 0
#define NNZ_TUPLES_COL
//#define NNZ_TUPLES_ROW

namespace autotuning {

using namespace combblas;

enum NNZ_STRAT {NNZ_GLOB_DENSITY, NNZ_LOC_DENSITY, NNZ_ARR};
enum SPLIT {COL_SPLIT, ROW_SPLIT}; //TODO: Move this into SpParMatInfo

template <typename IT, typename NT, typename DER>
class SpParMatInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;

    SpParMatInfo(){}

    //NOTE: loc* are values for the actual 2D processor grid
    //distInfo determines if distribution-specific information, like the array of tile densities, is computed
    SpParMatInfo(SpParMat3D<IT,NT,DER> * Mat):
        
        locMat(Mat->seqptr()),

        locNnz(Mat->seqptr()->getnnz()), 

        locNcols(Mat->getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat->getnrow() / RoundedSqrt<IT,IT>(worldSize)),
        locNcolsExact(Mat->seqptr()->getncol()),
        locNrowsExact(Mat->seqptr()->getnrow()),

        rowRank(Mat->getcommgrid()->GetCommGridLayer()->GetRankInProcRow()),
        colRank(Mat->getcommgrid()->GetCommGridLayer()->GetRankInProcCol())

    {
        SetGlobalInfo(Mat);
    }

    //TODO: Make a setlocalinfo function

    SpParMatInfo(SpParMat<IT,NT,DER>* Mat):

        locMat(Mat->seqptr()),
        locNnz(Mat->seqptr()->getnnz()), 
        locNcols(Mat->getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat->getnrow() / RoundedSqrt<IT,IT>(worldSize)),
        locNcolsExact(Mat->seqptr()->getncol()),
        locNrowsExact(Mat->seqptr()->getnrow()),

        rowRank(Mat->getcommgrid()->GetRankInProcRow()),
        colRank(Mat->getcommgrid()->GetRankInProcCol()),
        rank(Mat->getcommgrid()->GetRank()),
        nnz(Mat->getnnz()),
        ncols(Mat->getncol()),
        nrows(Mat->getnrow()),
        globDensity(static_cast<float>(Mat->getnnz()) / static_cast<float>(Mat->getncol()*Mat->getnrow()))

    {
    }

    
    template <typename M>
    void SetGlobalInfo(M * Mat) {
        this->nnz = Mat->getnnz();
        this->ncols = Mat->getncol();
        this->nrows = Mat->getnrow();
        this->globDensity = static_cast<float>(this->nnz) / static_cast<float>(this->ncols*this->nrows);
    }


    IT ComputeMsgSize(const int locNnz) {
        return locNnz * GetNzvalSize() +
                locNnz * GetIndexSize() +
                (locNnz + 1) * GetIndexSize();
    }


    // row, column
    void SetGridDims(SpGEMMParams& params) {

        /* Info about currently, actually formed 2D processor grid */
        const int totalProcs2D = jobPtr->totalTasks;
        const int procCols2D = RoundedSqrt<int,int>(totalProcs2D);
        const int procRows2D = procCols2D;

        /* Info about 3D grid */
        const int ppn = params.GetPPN();
        const int nodes = params.GetNodes();
        const int layers = params.GetLayers();
        const int totalProcs = params.GetTotalProcs();
        const int gridSize = params.GetGridSize();
        const int gridRows = params.GetGridDim();
        const int gridCols = gridRows;

        IT locNcolsGrid;
        IT locNrowsGrid;

        if (split==COL_SPLIT) {
            locNcolsGrid = locNcols * (procCols2D/gridCols)/layers;
            locNrowsGrid = locNrows * (procRows2D/gridRows);
        } else if (split==ROW_SPLIT) {
            locNcolsGrid = locNcols * (procCols2D/gridCols);
            locNrowsGrid = locNrows * (procRows2D/gridRows)/layers;
        } else {
            UNREACH_ERR();
        }

        gridDims = std::make_pair(locNrowsGrid, locNcolsGrid);

    }



    /* Vector of ranks in processor row procRank belongs to, including procRank */
    std::vector<int> RowRanks(const int procRank, const SpGEMMParams& params) {

        std::vector<int> ranks;
        ranks.reserve(params.GetGridDim()); 
        
        for (int p=0; p<params.GetGridDim(); p++) {
            int currRank = procRank / params.GetGridDim() + p;
            ranks.push_back(currRank);
        }
        
        return ranks;
        
    }


    /* Vector of ranks in processor column procRank belongs to, including procRank */
    std::vector<int> ColRanks(const int procRank, const SpGEMMParams& params) {

        std::vector<int> ranks;
        ranks.reserve(params.GetGridDim()); 
        
        for (int p=0; p<params.GetGridDim(); p++) {
            int currRank = procRank % params.GetGridDim() + (p*params.GetGridDim());
            ranks.push_back(currRank);
        }
        
        return ranks;
        
    }


    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}

    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}
    inline float GetGlobDensity() const {return globDensity;}

    inline IT GetLocNnz() const {return locNnz;}
    inline IT GetLocNcols() const {return locNcols;}
    inline IT GetLocNrows() const {return locNrows;}

    inline SPLIT GetSplit() const {return split;}

    inline std::pair<IT,IT> GetGridDims() {return gridDims;}

protected:

    // Global info
    IT nnz;
    IT ncols;
    IT nrows;
    float globDensity;

    // Info about actual 2D grid
    IT locNnz;
    IT locNcols;
    IT locNrows;
    IT locNcolsExact;
    IT locNrowsExact;
    int rowRank; //rank in actual 2d grid
    int colRank; // ||
    int rank;    // ||
    DER * locMat;

    // Row or column split
    SPLIT split;    

    // Dimensions of tile in hypothetical 3D grid
    std::pair<IT,IT> gridDims;
    
};


}//autotuning


#endif
