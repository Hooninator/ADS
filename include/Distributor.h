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
                                    IT ncols, IT nrows,
                                    std::shared_ptr<CommGrid> oldGrid,
                                    SpGEMMParams& oldParams,
                                    std::shared_ptr<CommGrid> newGrid,
                                    SpGEMMParams& newParams)
    {
        /* First, create mapping array */

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

        /* Setup send buffer */

        std::vector<std::vector<std::tuple<IT,IT,NT>>> sendBufs(worldSize);
        if (oldGrid != nullptr) {
            for (auto colIter = Mat->begcol(); colIter != Mat->endcol(); colIter++) {
                for (auto nzIter = Mat->begnz(colIter); nzIter != Mat->endnz(colIter); nzIter++) {
                    int targetRankInNew = TargetRankMapper(nzIter.rowid(), 
                                                            colIter.colid(), 
                                                            ncols, nrows,
                                                            Mat->GetGridRows());
                    int targetRank = newToWorldMap[targetRankInNew];

                    sendBufs[targetRank]->push_back({nzIter.rowid(), 
                                                     colIter.colid(),
                                                     nzIter.value()});

                }
            }
        }

        /* Setup Alltoallv */

        std::vector<std::tuple<IT,IT,NT>> sendBuf;
        sendBuf.reserve(Mat->getnnz());

        std::vector<int> sendDispls(worldSize);
        std::vector<int> sendSizes(worldSize);
        IT displ = 0;

        /* Send displs and sizes */
        for (int i=0; i<worldSize; i++) {

            sendDispls[i] = displ;

            sendBuf.insert(sendBuf.end(), sendBufs[i].begin(), sendBufs[i].end());

            displ += sendBufs[i].size();

            sendSizes[i] = sendBufs[i].size();

        }

        /* Recv sizes and displs */

        std::vector<int> recvSizes(worldSize);

        MPI_Alltoall((void*)(sendSizes.data()), 1, MPI_INT, (void*)(recvSizes.data()), 1,MPI_INT , MPI_COMM_WORLD);

        std::vector<IT> recvDispls(worldSize);
        recvDispls[0] = IT(0);

        for (int i=1; i<recvDispls.size(); i++) {
            recvDispls[i] = recvDispls[i-1] + recvSizes[i-1];
        }

        std::vector<std::tuple<IT,IT,NT>> recvBuf(*(recvDispls.end()));

        /* Alltoallv */
        MPI_Alltoallv((void*)(sendBuf.data()), sendSizes.data(), sendDispls.data(), MPIType<std::tuple<IT,IT,NT>>(), 
                        (void*)(recvBuf.data()), recvSizes.data(), recvDispls.data(),MPIType<std::tuple<IT,IT,NT>>(),
                        MPI_COMM_WORLD);

        /* Make new local matrix */
        auto newLocDims = GetLocDim(ncols, nrows, newGrid);

        SpTuples<IT,NT> * tuples = new SpTuples(recvBuf.size(), 
                                                std::get<0>(newLocDims),
                                                std::get<1>(newLocDims),
                                                recvBuf.data());

        return new DER(std::get<0>(newLocDims), std::get<1>(newLocDims),
                        tuples.size(), tuples, false);
                                                    

    }


    template <typename IT>
    static int TargetRankMapper(IT i, IT j, 
                                IT ncols, IT nrows,
                                int newGridDim)
    {
        
        IT locCols = ncols / newGridDim;
        IT locRows = nrows / newGridDim;

        int targetRank = (i / locRows) * newGridDim + (j / locCols);

        return targetRank;
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
