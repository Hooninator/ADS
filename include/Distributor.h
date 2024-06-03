#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

#include "common.h"
#include "SpGEMMParams.h"

namespace autotuning {
using namespace combblas;



class Distributor
{

public:

    Distributor() {}

    
    template <typename IT, typename NT, typename DER>
    static DER * ReDistributeMatrix(SpMat<IT,NT,DER> * Mat,
                                    std::shared_ptr<CommGrid> oldGrid,
                                    SpGEMMParams& oldParams,
                                    std::shared_ptr<CommGrid> newGrid,
                                    SpGEMMParams& newParams)
    {
        bool isScalingDown = (oldParams.GetTotalProcs() > newParams.GetTotalProcs());

        /* Allreduce to get complete matrix dimensions */

        IT buf[2]; 

        if (Mat != nullptr) {
            buf[0] = Mat->getnrow();
            buf[1] = Mat->getncol();
        } else {
            std::memset(buf, 0, sizeof(IT)*2);
        }

        MPI_Allreduce(MPI_IN_PLACE, (void*)(buf), 2, MPIType<IT>(), MPI_SUM,
                        MPI_COMM_WORLD);

        IT ncols, nrows;
        nrows = buf[0] / oldParams.GetGridDim();
        ncols = buf[1] / oldParams.GetGridDim();

#ifdef DEBUG
        debugPtr->Print("Done with dimension allreduce");
#endif

        /* Create mapping array */

        std::vector<int> newToWorldMap(worldSize);

        if (newGrid != nullptr) {
            int rankInNew = newGrid->GetRank();
            newToWorldMap[rankInNew] = rank; //world rank
        }

        MPI_Allreduce(MPI_IN_PLACE, (void*)(newToWorldMap.data()), 
                        newToWorldMap.size(),
                        MPI_INT,
                        MPI_SUM,
                        MPI_COMM_WORLD);
#ifdef DEBUG
        debugPtr->Print("Done with map array allreduce");
#endif

        /* Setup send buffer */

        std::vector<std::vector<std::tuple<IT,IT,NT>>> sendBufs(worldSize);
        if (oldGrid != nullptr) {

            IT locRowsOld = nrows / oldParams.GetGridDim();
            IT locColsOld = ncols / oldParams.GetGridDim();
            IT locRowsNew = nrows / newParams.GetGridDim();
            IT locColsNew = ncols / newParams.GetGridDim();

            IT rowOffset, colOffset;
            if (isScalingDown) {
                int rowSizeSuperTile = oldParams.GetGridDim() / newParams.GetGridDim();
                rowOffset = (oldGrid->GetRankInProcCol() % rowSizeSuperTile) * locRowsOld;
                colOffset = (oldGrid->GetRankInProcRow() % rowSizeSuperTile) * locColsOld;
            } else {
                rowOffset = 0;
                colOffset = 0;
            }
#ifdef DEBUG
            debugPtr->Log("Row offset " + std::to_string(rowOffset));
            debugPtr->Log("Col offset " + std::to_string(colOffset));
#endif


#ifdef DEBUG
            debugPtr->Print("Rank " + std::to_string(rank));
            debugPtr->Print(Mat->getncol());
            debugPtr->Print(Mat->getnrow());
#endif

            for (auto colIter = Mat->begcol(); colIter != Mat->endcol(); colIter++) {
                for (auto nzIter = Mat->begnz(colIter); nzIter != Mat->endnz(colIter); nzIter++) {

                    IT rowOffsetGlobal = 0;
                    IT colOffsetGlobal = 0;

                    rowOffsetGlobal = oldGrid->GetRankInProcCol() * locRowsOld;
                    colOffsetGlobal = oldGrid->GetRankInProcRow() * locColsOld;

                    auto targetRanksInNew = TargetRankMapper(nzIter.rowid() + rowOffsetGlobal,
                                                            colIter.colid() + colOffsetGlobal, 
                                                            ncols, nrows,
                                                            newParams.GetGridDim());

                    int rowRankInNew = std::get<0>(targetRanksInNew);
                    int colRankInNew = std::get<1>(targetRanksInNew);
                    int targetRankInNew = colRankInNew * newParams.GetGridDim() + rowRankInNew;

                    int targetRank = newToWorldMap[targetRankInNew];


#ifdef DEBUG
                    std::stringstream ss;
                    ss<<"("<<nzIter.rowid() + rowOffset <<","<<colIter.colid() + colOffset <<"): "<<targetRankInNew<<","<<targetRank<<std::endl;
                    debugPtr->Log(ss.str());
#endif

                    if (!isScalingDown) {
                        int superRowSize = newParams.GetGridDim() / oldParams.GetGridDim();
                        IT rowOnset = locRowsNew * (colRankInNew % superRowSize) * -1;
                        IT colOnset = locColsNew * (rowRankInNew % superRowSize) * -1;
                        sendBufs[targetRank].push_back({nzIter.rowid() + rowOnset, 
                                                         colIter.colid() + colOnset,
                                                         nzIter.value()});
                    } else {
                        sendBufs[targetRank].push_back({nzIter.rowid() + rowOffset, 
                                                         colIter.colid() + colOffset,
                                                         nzIter.value()});
                    }

                }
            }
        }

#ifdef DEBUG
        debugPtr->Print("Done with send buffer setup");
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        /* Setup Alltoallv */

        std::vector<std::tuple<IT,IT,NT>> sendBuf;
        sendBuf.reserve(Mat->getnnz());

        std::vector<int> sendDispls(worldSize);
        std::vector<int> sendSizes(worldSize);
        IT displ = 0;

        /* Send displs and sizes */
        for (int i=0; i<worldSize; i++) {

            sendBuf.insert(sendBuf.end(), sendBufs[i].begin(), sendBufs[i].end());

            sendDispls[i] = displ;
            displ += sendBufs[i].size();

            sendSizes[i] = sendBufs[i].size();

        }

#ifdef DEBUG
        debugPtr->Print("Done with send displ and sizes");
        debugPtr->LogVecSameLine(sendDispls, "sendDispls");
        debugPtr->LogVecSameLine(sendSizes, "sendSizes");
#endif

        /* Recv sizes and displs */

        std::vector<int> recvSizes(worldSize);

        MPI_Alltoall((void*)(sendSizes.data()), 1, MPI_INT, (void*)(recvSizes.data()), 1,MPI_INT , MPI_COMM_WORLD);


        std::vector<int> recvDispls(worldSize);
        recvDispls[0] = 0;

        for (int i=1; i<recvDispls.size(); i++) {
            recvDispls[i] = recvDispls[i-1] + recvSizes[i-1];
        }

#ifdef DEBUG
        debugPtr->Print("Send sizes alltoall done");
        debugPtr->LogVecSameLine(recvSizes, "recvSizes");
        debugPtr->LogVecSameLine(recvDispls, "recvDispls");
#endif

        std::vector<std::tuple<IT,IT,NT>> recvBuf(std::reduce(recvSizes.begin(), recvSizes.end(), 0));

        /* Alltoallv */
        MPI_Alltoallv((void*)(sendBuf.data()), sendSizes.data(), sendDispls.data(), MPIType<std::tuple<IT,IT,NT>>(), 
                        (void*)(recvBuf.data()), recvSizes.data(), recvDispls.data(),MPIType<std::tuple<IT,IT,NT>>(),
                        MPI_COMM_WORLD);
#ifdef DEBUG
        debugPtr->Print("alltoallv done");
#endif

        /* Make new local matrix */
        auto newLocDims = GetLocDim(ncols, nrows, newGrid);
#ifdef DEBUG
        debugPtr->Log("NewLocDims " + std::to_string(std::get<0>(newLocDims)) + ", " + std::to_string(std::get<1>(newLocDims)));
#endif

        bool isSorted = isScalingDown;

        SpTuples<IT,NT> * tuples = new SpTuples(recvBuf.size(), 
                                                std::get<0>(newLocDims),
                                                std::get<1>(newLocDims),
                                                recvBuf.data(),
                                                isSorted, false);
#ifdef DEBUG
        debugPtr->Print("Matrix construction done");
#endif

        return new DER(*tuples, false);
                                                    

    }


    template <typename IT>
    static std::tuple<int,int> TargetRankMapper(IT i, IT j, 
                                                IT ncols, IT nrows,
                                                int newGridDim)
    {
        
        IT locCols = ncols / newGridDim;
        IT locRows = nrows / newGridDim;


        return {j / locCols, i / locRows};
    }

    
    template <typename IT>
    static std::tuple<IT,IT> GetLocDim(IT ncols, IT nrows,
                                        std::shared_ptr<CommGrid>& newGrid)
    {
        if (newGrid == nullptr) return std::tuple<IT,IT>{0,0};

        int gridDim = newGrid->GetGridRows();

        IT locCols = ncols / gridDim;
        IT locRows = nrows / gridDim;

        /* Check for edge case row/column */
        int rowRank = newGrid->GetRankInProcRow();
        int colRank = newGrid->GetRankInProcCol();
        
        if ((rowRank + 1) == gridDim) {
            IT edgeColsSize = ( std::abs( std::ceil((double) ncols / (double) gridDim) -
                                          locCols) ); 
            locCols += edgeColsSize;
        }

        if ((colRank + 1) == gridDim) {
            IT edgeRowsSize = ( std::abs( std::ceil((double) nrows / (double) gridDim) -
                                          locRows) ); 
            locRows += edgeRowsSize;
        }


        return std::tuple<IT, IT>{locRows, locCols};
    }

};




}


#endif
