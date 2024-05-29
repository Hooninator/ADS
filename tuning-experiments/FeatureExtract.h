
#ifndef FEATURE_EXTRACT_H
#define FEATURE_EXTRACT_H


#include "common.h"

#define PRECISION 20

using namespace combblas;


template <typename IT, typename NT, typename DER>
class FeatureExtractor {

typedef std::map<std::string, std::string> Map;

public:

FeatureExtractor(){}


void MakeSample2D(SpParMat<IT,NT,DER>& A, SpParMat<IT,NT,DER>& B, Map * timings, std::ofstream& ofs) {

    int rank = A.getcommgrid()->GetRank();
    if (rank==0)
        std::cout<<"Making sample..."<<std::endl;
    
    Map * featMap = new Map();

    std::string Astr("A");
    std::string Bstr("B");

    MPI_Barrier(MPI_COMM_WORLD);

    // Job stats
    JobStats(featMap);

    // Global matrix stats
    GlobStats(A, featMap, Astr);
    GlobStats(B, featMap, Bstr);
    if (rank==0)
        std::cout<<"Extracted global stats..."<<std::endl;

    // Per column stats 
    ColNnzStats(A, featMap, Astr);
    ColNnzStats(B, featMap, Bstr);
    if (rank==0)
        std::cout<<"Extracted column stats..."<<std::endl;
    
    // Tile stats
    TileNnzStats(A, featMap, Astr);
    TileNnzStats(B, featMap, Bstr);
    if (rank==0)
        std::cout<<"Extracted tile stats..."<<std::endl;

    // Tile col stats
    TileColNnzStats(A, featMap, Astr);
    TileColNnzStats(B, featMap, Bstr);
    if (rank==0)
        std::cout<<"Extracted tile col stats..."<<std::endl;

    // FLOP stats
    typedef PlusTimesSRing<NT,NT> PTTF;
    IT localFLOPS, globalFLOPS = 0;
    globalFLOPS = EstimateFLOP<PTTF, IT, NT, NT, DER, DER>(A, B, false, false, &localFLOPS);

    float avgFLOPS, stdDevFLOPS = 0;
    IT minFLOPS, maxFLOPS = 0;
    
    avgFLOPS = static_cast<float>(globalFLOPS) / static_cast<float>(A.getcommgrid()->GetSize());

    MPI_Reduce((void*)(&localFLOPS), (void*)(&minFLOPS), 1, MPIType<IT>(), MPI_MIN, 0, A.getcommgrid()->GetWorld());
    MPI_Reduce((void*)(&localFLOPS), (void*)(&maxFLOPS), 1, MPIType<IT>(), MPI_MAX, 0, A.getcommgrid()->GetWorld());
    
    stdDevFLOPS = std::pow( localFLOPS - avgFLOPS, 2 );
    MPI_Allreduce(MPI_IN_PLACE, (void*)(&stdDevFLOPS), 1, MPI_FLOAT, MPI_SUM, A.getcommgrid()->GetWorld());
    stdDevFLOPS = std::sqrt(stdDevFLOPS / static_cast<float>(A.getcommgrid()->GetSize())); 

    if (rank==0) {
        featMap->emplace("avg-FLOPS", STR(avgFLOPS));
        featMap->emplace("min-FLOPS", STR(minFLOPS));
        featMap->emplace("max-FLOPS", STR(maxFLOPS));
        featMap->emplace("stdev-FLOPS", STR(stdDevFLOPS));
        featMap->emplace("global-FLOPS", STR(globalFLOPS));
        std::cout<<"Extracted FLOP stats..."<<std::endl;
    }


    if (rank==0) {
        WriteSample(featMap, timings, ofs);
        std::cout<<"Wrote sample..."<<std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

}

/* Features:
 *     nnz-{A,B}
 *     FLOPS
 *     m-{A,B}
 *     n-{A,B}
 *     outputNnz-intermediate
 *     outputNnz-final
 * rank is just used to construct the graph
 */

void MakeSampleGNN(SpParMat<IT,NT,DER>& A, SpParMat<IT,NT,DER>& B, Map * timings, std::ofstream& ofs) {


    int rank = A.getcommgrid()->GetRank();
    
    if (rank==0) ofs<<"----SAMPLE----"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);


    Map * featMap = new Map();

    featMap->emplace("rank", STR(rank));

    featMap->emplace("nnz-A", STR(A.seqptr()->getnnz()));
    featMap->emplace("nnz-B", STR(B.seqptr()->getnnz()));

    typedef PlusTimesSRing<NT,NT> PTTF;
    IT localFLOPS = 0;
    EstimateFLOP<PTTF, IT, NT, NT, DER, DER>(A, B, false, false, &localFLOPS);
    featMap->emplace("FLOPS", STR(localFLOPS));

    featMap->emplace("m-A", STR(A.seqptr()->getnrow()));
    featMap->emplace("m-B", STR(B.seqptr()->getnrow()));
    featMap->emplace("n-A", STR(A.seqptr()->getncol()));
    featMap->emplace("n-B", STR(B.seqptr()->getncol()));

    IT * flopC = estimateFLOP(*(A.seqptr()), *(B.seqptr()));
    
    IT outputNnz = 0;
    if (!(A.seqptr()->isZero()) && !(B.seqptr()->isZero())) {
        IT * outputNnzCol = estimateNNZ_Hash(*(A.seqptr()), *(B.seqptr()), flopC);
        for (int i=0; i<B.seqptr()->GetDCSC()->nzc; i++)
        {
            outputNnz += outputNnzCol[i];
        }
    }
    featMap->emplace("outputNnz-intermediate", STR(outputNnz));

    int64_t outputNnzFinal = EstPerProcessNnzSUMMAMax(A,B, false);
    featMap->emplace("outputNnz-final", STR(outputNnzFinal));

    WriteSample(featMap, timings, ofs);
    std::cout<<"Wrote sample!"<<std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
}

void WriteSample(const Map * features, const Map * timings, std::ofstream& ofs) {

    // Write features
    std::for_each(features->begin(), features->end(),
        [&ofs](auto const& elem) {
            ofs<<elem.first<<":"<<elem.second<<" ";
        }
    );

    // Write timings
    std::for_each(timings->begin(), timings->end(),
        [&ofs](auto const& elem) {
            ofs<<elem.first<<":"<<elem.second<<" ";
        }
    );

    // Newline, end of feature
    ofs<<std::endl;

}


template <typename T>
std::string ToStrScientific(T input) {
    std::ostringstream ss;
    ss<<std::scientific<<std::setprecision(PRECISION)<<input;
    return ss.str();
}




void JobStats(Map * featMap) {
    featMap->emplace("Nodes", std::getenv("SLURM_NNODES"));
    featMap->emplace("PPN", std::getenv("SLURM_NTASKS_PER_NODE"));
}


void GlobStats(SpParMat<IT,NT,DER>& Mat, Map * featMap, std::string& matSymb) {
    int rank = Mat.getcommgrid()->GetRank();
    IT nnz, nrow, ncol;
    nnz = Mat.getnnz();
    nrow = Mat.getnrow();
    ncol = Mat.getncol();
    if (rank==0) { 
        featMap->emplace("nnz-"+matSymb, ToStrScientific(nnz));
        featMap->emplace("m-"+matSymb, ToStrScientific(nrow));
        featMap->emplace("n-"+matSymb, ToStrScientific(ncol));
        featMap->emplace("density-"+matSymb, 
                            ToStrScientific(static_cast<float>(nnz) / static_cast<float>(ncol * nrow) ) );
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


// avg nnz, min, max, stdev per col
void ColNnzStats(SpParMat<IT,NT,DER>& Mat, Map * featMap, std::string& matSymb) {

    int rank = Mat.getcommgrid()->GetRank();

    // Average nnz per column
    double avgNnz = static_cast<double>(Mat.getnnz()) / static_cast<double>(Mat.getncol());
    
    // Min and max nnz per col -- requires communication since matrix is distributed
    // Populate array with local nnz counts
    std::vector<IT> locNnzColArr(Mat.seqptr()->getncol());
    for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
        locNnzColArr[colIter.colid()] = colIter.nnz();
    }

    // Allreduce along each processor column
    MPI_Allreduce(MPI_IN_PLACE, (void*)(locNnzColArr.data()), locNnzColArr.size(),
                    MPIType<IT>(), MPI_SUM, Mat.getcommgrid()->GetColWorld());

    //Setup gatherv
    std::vector<int> recvCounts(Mat.getcommgrid()->GetGridCols());
    int locNcols = Mat.seqptr()->getncol();
    MPI_Allgather((void*)(&locNcols), 1, MPI_INT, (void*)(recvCounts.data()), 1, MPI_INT, 
                    Mat.getcommgrid()->GetRowWorld());

    std::vector<int> displs(Mat.getcommgrid()->GetGridCols());
    for (int i=1; i<displs.size(); i++) {
        displs[i] = displs[i-1] + recvCounts[i-1];
    }

    // Gatherv along each processor row to get global col nnz
    // NOTE: Gatherv is needec because the last processor column can have edge case columns
    std::vector<IT> globNnzColArr(Mat.getncol());
    MPI_Gatherv((void*)(locNnzColArr.data()), locNnzColArr.size(), MPIType<IT>(),
                    (void*)(globNnzColArr.data()), recvCounts.data(), displs.data(), MPIType<IT>(),
                    0, Mat.getcommgrid()->GetRowWorld());

    IT nrows = Mat.getnrow();
    IT ncols = Mat.getnrow();

    if (rank==0) {
        // Reduce to get min and max
        IT minNnzCol, maxNnzCol;
        minNnzCol = ReduceMin(globNnzColArr);
        maxNnzCol = ReduceMax(globNnzColArr);

        // Reduce to get stdev
        double stdev;
        stdev = ReduceStdev(globNnzColArr, avgNnz);

        // Compute density info
        double density, avgDensity, minDensity, maxDensity, stdevDensity;
        std::vector<double> colDensities(ncols);
        std::transform(globNnzColArr.begin(), globNnzColArr.end(), colDensities.begin(),
                        [&nrows](IT colNnz) {return static_cast<double>(colNnz)/static_cast<double>(nrows);});

        avgDensity = ReduceMean(colDensities);
        minDensity = ReduceMin(colDensities);
        maxDensity = ReduceMax(colDensities);
        stdevDensity = ReduceStdev(colDensities, avgDensity);

        // Write values to featMap
        featMap->emplace("minNnzCol-"+matSymb, ToStrScientific(minNnzCol));
        featMap->emplace("maxNnzCol-"+matSymb, ToStrScientific(maxNnzCol));
        featMap->emplace("stdevNnzCol-"+matSymb, ToStrScientific(stdev));
        featMap->emplace("avgNnzCol-"+matSymb, ToStrScientific(avgNnz));
        featMap->emplace("avgDensityCol-"+matSymb, ToStrScientific(avgDensity));
        featMap->emplace("minDensityCol-"+matSymb, ToStrScientific(minDensity));
        featMap->emplace("maxDensityCol-"+matSymb, ToStrScientific(maxDensity));
        featMap->emplace("stdevDensityCol-"+matSymb, ToStrScientific(stdevDensity));
    }

    MPI_Barrier(MPI_COMM_WORLD);

}

//Average, min, max, stdev, density for nnz per tile
void TileNnzStats(SpParMat<IT,NT,DER>& Mat, Map * featMap, std::string& matSymb) {

    int rank = Mat.getcommgrid()->GetRank();

    double avgNnzTile = (static_cast<double>(Mat.getnnz()) / static_cast<double>(Mat.getcommgrid()->GetSize()) );

    IT locNrows = Mat.seqptr()->getnrow();
    IT locNcols = Mat.seqptr()->getncol();

    double density = static_cast<double>(Mat.seqptr()->getnnz()) / 
                        static_cast<long double>(Mat.seqptr()->getncol() * Mat.seqptr()->getnrow());
    std::vector<double> tileDensities(Mat.getcommgrid()->GetSize());
    MPI_Allgather((void*)(&density), 1, MPI_DOUBLE, (void*)(tileDensities.data()), 1, MPI_DOUBLE, Mat.getcommgrid()->GetWorld());

    double avgDensity = ReduceMean(tileDensities);
    double minDensity = ReduceMin(tileDensities);
    double maxDensity = ReduceMax(tileDensities);
    double stdevDensity = ReduceStdev(tileDensities, avgDensity);

    std::vector<IT> globNnzTile(Mat.getcommgrid()->GetSize());
    IT locNnzTile = Mat.seqptr()->getnnz();

    // Gather to get complete list of tile nnz counts
    MPI_Gather((void*)(&locNnzTile), 1, MPIType<IT>(), 
                     (void*)(globNnzTile.data()), 1, MPIType<IT>(), 
                        0, Mat.getcommgrid()->GetWorld());
    
    if (rank==0) {
        // Get min and max nnz
        IT minNnzTile, maxNnzTile;
        minNnzTile = ReduceMin(globNnzTile);
        maxNnzTile = ReduceMax(globNnzTile);

        // Reduce to get stdev
        double stdev;
        stdev = ReduceStdev(globNnzTile, avgNnzTile);

        // Write values to featMap
        featMap->emplace("mTile-"+matSymb, ToStrScientific(locNrows));
        featMap->emplace("nTile-"+matSymb, ToStrScientific(locNcols));
        featMap->emplace("minNnzTile-"+matSymb, ToStrScientific(minNnzTile));
        featMap->emplace("maxNnzTile-"+matSymb, ToStrScientific(maxNnzTile));
        featMap->emplace("stdevNnzTile-"+matSymb, ToStrScientific(stdev));
        featMap->emplace("avgNnzTile-"+matSymb, ToStrScientific(avgNnzTile));
        featMap->emplace("avgDensityTile-"+matSymb, ToStrScientific(avgDensity));
        featMap->emplace("minDensityTile-"+matSymb, ToStrScientific(minDensity));
        featMap->emplace("maxDensityTile-"+matSymb, ToStrScientific(maxDensity));
        featMap->emplace("stdevDensityTile-"+matSymb, ToStrScientific(stdevDensity));
    }

    MPI_Barrier(MPI_COMM_WORLD);

}


void TileColNnzStats(SpParMat<IT,NT,DER>& Mat, Map * featMap, std::string& matSymb) {

    int rank = Mat.getcommgrid()->GetRank();

    IT n = Mat.getncol() * Mat.getcommgrid()->GetGridRows();

    double avgNnzTileCol = static_cast<double>(Mat.getnnz()) / static_cast<double>(n);
    
    // Local tile col nnz
    std::vector<IT> * locTileColNnz = new std::vector<IT>(Mat.seqptr()->getncol());
    for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
        locTileColNnz->at(colIter.colid()) = colIter.nnz();
    }


    // Setup gatherv
    std::vector<int> * recvCounts = new std::vector<int>(Mat.getcommgrid()->GetGridCols());
    int locNcols = Mat.seqptr()->getncol();
    MPI_Allgather((void*)(&locNcols), 1, MPI_INT, (void*)(recvCounts->data()), 1, MPI_INT, Mat.getcommgrid()->GetRowWorld()); 

    std::vector<int> * displs = new std::vector<int>(Mat.getcommgrid()->GetGridCols());
    for (int i=1; i<displs->size(); i++) {
        displs->at(i) = displs->at(i-1) + recvCounts->at(i-1);
    }

    //if (rank==0) std::cout<<"Starting gather"<<std::endl;
    // Aggregate all tile col nnz on processor column 0
    std::vector<IT> * rowWorldTileColNnz = new std::vector<IT>(Mat.getncol());
    MPI_Gatherv((void*)(locTileColNnz->data()), locTileColNnz->size(), MPIType<IT>(),
                (void*)(rowWorldTileColNnz->data()), recvCounts->data(), displs->data(), MPIType<IT>(),
                0, Mat.getcommgrid()->GetRowWorld());
    //if (rank==0) std::cout<<"Done with gather"<<std::endl;

    // Avoid creating large n*ncols arrays
    IT locNrows = Mat.seqptr()->getnrow();
    IT ncols = Mat.getncol();
    if (Mat.getcommgrid()->GetRankInProcRow()==0) {

        // Local operations 
        IT locMinNnzTileCol = ReduceMin(*rowWorldTileColNnz);
        IT locMaxNnzTileCol = ReduceMax(*rowWorldTileColNnz);
        long double locStdevContrib = std::pow(ReduceStdev(*rowWorldTileColNnz, avgNnzTileCol) * rowWorldTileColNnz->size(), 2);

        // Reduce among processor column 0 to get complete values on rank 0
        IT minNnzTileCol, maxNnzTileCol;
        long double stdev;
        MPI_Reduce((void*)(&locMinNnzTileCol), (void*)(&minNnzTileCol), 1, MPIType<IT>(),
                    MPI_MIN, 0, Mat.getcommgrid()->GetColWorld());
        MPI_Reduce((void*)(&locMaxNnzTileCol), (void*)(&maxNnzTileCol), 1, MPIType<IT>(),
                    MPI_MAX, 0, Mat.getcommgrid()->GetColWorld());
        MPI_Reduce((void*)(&locStdevContrib), (void*)(&stdev), 1, MPI_LONG_DOUBLE,
                    MPI_SUM, 0, Mat.getcommgrid()->GetColWorld());
        stdev = std::sqrt(stdev) / static_cast<double>(n);

        // Compute density info
        double locDensitySum, locMinDensity, locMaxDensity, locStdevDensityContrib;
        double avgDensity, minDensity, maxDensity, stdevDensity;
        std::vector<double> * tileColDensities = new std::vector<double>(ncols);
        std::transform(rowWorldTileColNnz->begin(), rowWorldTileColNnz->end(), tileColDensities->begin(),
            [&locNrows](IT tileColNnz){return static_cast<double>(tileColNnz) / static_cast<double>(locNrows);}); 
        
        locMinDensity = ReduceMin(*tileColDensities);
        locMaxDensity = ReduceMax(*tileColDensities);
        MPI_Reduce((void*)(&locMinDensity), (void*)(&minDensity), 1, MPI_DOUBLE, MPI_MIN, 0, Mat.getcommgrid()->GetColWorld());
        MPI_Reduce((void*)(&locMaxDensity), (void*)(&maxDensity), 1, MPI_DOUBLE, MPI_MAX, 0, Mat.getcommgrid()->GetColWorld());

        // sum all densities
        locDensitySum = std::reduce(tileColDensities->begin(), tileColDensities->end(), 0);
        MPI_Allreduce(MPI_IN_PLACE, (void*)(&locDensitySum), 1, MPI_DOUBLE, MPI_SUM, Mat.getcommgrid()->GetColWorld());

        // Compute avg
        avgDensity = locDensitySum / static_cast<double>(n);


        // Finally, compute stdev
        locStdevDensityContrib = std::pow(ReduceStdev(*tileColDensities, avgDensity) * tileColDensities->size(), 2);
        MPI_Reduce((void*)(&locStdevDensityContrib), (void*)(&stdevDensity), 1, MPI_DOUBLE, MPI_SUM, 0, Mat.getcommgrid()->GetColWorld());
        stdevDensity = std::sqrt(stdev) / static_cast<double>(n);

        if (rank==0) {
            // Write values to featMap
            featMap->emplace("minNnzTileCol-"+matSymb, ToStrScientific(minNnzTileCol));
            featMap->emplace("maxNnzTileCol-"+matSymb, ToStrScientific(maxNnzTileCol));
            featMap->emplace("stdevNnzTileCol-"+matSymb, ToStrScientific(stdev));
            featMap->emplace("avgNnzTileCol-"+matSymb, ToStrScientific(avgNnzTileCol));
            featMap->emplace("avgDensityTileCol-"+matSymb, ToStrScientific(avgDensity));
            featMap->emplace("minDensityTileCol-"+matSymb, ToStrScientific(minDensity));
            featMap->emplace("maxDensityTileCol-"+matSymb, ToStrScientific(maxDensity));
            featMap->emplace("stdevDensityTileCol-"+matSymb, ToStrScientific(stdevDensity));
        }

        delete tileColDensities;
    }

    delete locTileColNnz;
    delete rowWorldTileColNnz;
    delete recvCounts;
    delete displs;

    MPI_Barrier(MPI_COMM_WORLD);

}


template <typename T>
T ReduceMin(std::vector<T>& v) {
    return std::reduce(v.begin(), v.end(), T(0),
        [](T currMin, T curr) {return std::min(currMin, curr);}
    );
}


template <typename T>
T ReduceMax(std::vector<T>& v) {
    return std::reduce(v.begin(), v.end(), T(0),
        [](T currMax, T curr) {return std::max(currMax, curr);}
    );
}


template <typename T>
double ReduceStdev(std::vector<T>& v, double avg) {
    return std::sqrt(std::reduce(v.begin(), v.end(), 0.0,
        [&avg](double sum, T val) {
            return sum + (std::pow( static_cast<double>((val - avg)), 2)); 
        }
    ) / v.size());
}


template <typename T>
T ReduceMean(std::vector<T>& v) {
    return std::reduce(v.begin(), v.end(), T(0))/static_cast<T>(v.size());
}


};

#endif








