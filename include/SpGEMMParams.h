


#ifndef SPGEMMPARAMS_H
#define SPGEMMPARAMS_H

#include "common.h"
#include "PlatformParams.h"



namespace autotuning{

using namespace combblas;

class SpGEMMParams {
    
public:


	
    SpGEMMParams(){}


    SpGEMMParams(int nodes, int ppn, int layers):
        nodes(nodes), ppn(ppn), layers(layers),
        totalProcs(nodes*ppn)
    {
        gridSize = totalProcs / layers;
        gridDim = RoundedSqrt<int,int>(gridSize);
    }



    void Print() {
        std::cout<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
    }


    std::string OutStr() {
        std::stringstream ss;
        ss<<nodes<<","<<ppn<<","<<layers;
        return ss.str();
    }


    //TODO: Probably some smart way to make this more generic
    static std::vector<SpGEMMParams> ConstructSearchSpace3D(PlatformParams& params) {
        std::vector<SpGEMMParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=params.GetCoresPerNode(); _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes)) {
                    for (int _layers=1; _layers<=_ppn*_nodes; _layers*=2) {
                        int gridSize = (_ppn*_nodes) / _layers;
                        if (IsPerfectSquare(gridSize))
                            result.push_back(SpGEMMParams(_nodes,_ppn,_layers));
                    }
                }
            }
        }
        return result;
    }


    static std::vector<SpGEMMParams> ConstructSearchSpace2D(PlatformParams& params, uint32_t nodeLimit, uint32_t ppnLimit) {
        std::vector<SpGEMMParams> space;
        for (uint32_t _nodes = 1; _nodes<=nodeLimit; _nodes*=2) {
            for (uint32_t _ppn=1; _ppn<=ppnLimit; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes) ) {
                    space.push_back(SpGEMMParams(_nodes,_ppn,1));
                }
            }
        }

        return space;

    }

    
    static SpGEMMParams GetDefaultParams()
    {
        return SpGEMMParams(jobPtr->nodes, jobPtr->tasksPerNode, 1);
    }


    //TODO: I think we should split this into SpGEMM2DParams and SpGEMM3DParams...
    std::shared_ptr<CommGrid> MakeGridFromParams() 
    {

        std::vector<int> ranks; // Ranks to include in the new communicator
        ranks.reserve(nodes*ppn);
        
        // Get all ranks within this parameter object's nodes and ppn
        for (int rank=0; rank<autotuning::worldSize; rank++)
        {
            int nodeRank = rank / jobPtr->tasksPerNode;
            int localRank = rank % jobPtr->tasksPerNode;
            if (nodeRank < nodes && localRank < ppn)
            {
                ranks.push_back(rank);
            }
        }

        // Make the new group 
        MPI_Group worldGroup;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

        MPI_Group newGroup;
        MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &newGroup);

        MPI_Comm newComm;
        MPI_Comm_create_group(MPI_COMM_WORLD, newGroup, 0, &newComm);

        int newSize;
        MPI_Comm_size(newComm, &newSize);

        if (newComm != MPI_COMM_NULL) {

            ASSERT(newSize==nodes*ppn,
            "Expected communicator size to be " + std::to_string(nodes*ppn) + ", but it was "
            + std::to_string(newSize));

            ASSERT(IsPerfectSquare(newSize / layers),
            "Each 2D grid must be a perfect square, instead got " + OutStr());

            std::shared_ptr<CommGrid> newGrid;
            newGrid.reset(new CommGrid(newComm, 0, 0));

            return newGrid;

        } else {
            return NULL;
        }

    }


    //TODO: This should be moved to a separate class
    template <typename DER>
    DER * ReDistributeSpMat(DER * mat, SpGEMMParams& oldParams)
    {

        DER *newMat;
        
        auto newParams = *(this);

        auto isInNewGrid = [&newParams](int rank )
        {
            int nodeRank = rank / jobPtr->tasksPerNode;
            int localRank = rank % jobPtr->tasksPerNode;
            return nodeRank < newParams.GetNodes() && localRank < newParams.GetPPN();
        };

        auto isInOldGrid = [&oldParams](int rank)
        {
            int nodeRank = rank / jobPtr->tasksPerNode;
            int localRank = rank % jobPtr->tasksPerNode;
            return nodeRank < oldParams.GetNodes() && localRank < oldParams.GetPPN();
        };

        // Given the rank of a process in COMM_WORLD, get the rank of that process in old grid
        std::map<int, int> worldToOldGridRank;
        // Given the rank of a process in old grid, get the rank of that procss in COMM_WORLD
        std::map<int, int> oldGridToWorldRank;

        // Given the rank of a process in COMM_WORLD, get the rank of that process in new grid
        std::map<int, int> worldToNewGridRank;
        // Given the rank of a process in new grid, get the rank of that process in COMM_WORLD 
        std::map<int, int> newGridToWorldRank;

        int i=0;
        for (int node=0; node<newParams.nodes; node++) 
        {
            for (int p=0; p<newParams.ppn; p++) 
            {
                newGridToWorldRank[i] = p + node * newParams.ppn;
                worldToNewGridRank[p + node * newParams.ppn] = i;
                i++;
            }
        }

        i = 0;
        for (int node=0; node<oldParams.nodes; node++) 
        {
            for (int p=0; p<oldParams.ppn; p++) 
            {
                oldGridToWorldRank[i] = p + node * oldParams.ppn;
                worldToOldGridRank[ p + node * oldParams.ppn] = i;
                i++;
            }
        }



        int oldGridDim = RoundedSqrt<int,int>(oldParams.totalProcs);
        int oldGridSize = oldGridDim * oldGridDim;
        int rankInOldGridRow = worldToOldGridRank[rank] % oldGridDim;
        int rankInOldGridCol = worldToOldGridRank[rank] / oldGridDim;

        int newGridDim = RoundedSqrt<int,int>(newParams.totalProcs);
        int newGridSize = newGridDim * newGridDim;
        int rankInNewGridRow = worldToNewGridRank[rank] % newGridDim;
        int rankInNewGridCol = worldToNewGridRank[rank] / newGridDim;

        auto GetSuperTileIdx = [](int rowRank, int colRank,
                                int superRows, int superCols)
        { 
            return (rowRank / superCols)  + (colRank / superRows) * ( superRows);
        };

        auto GetRankInSuperTile = [](int gridRank, int superDim)
        {
            return gridRank % superDim;
        };
     

        // Same size. Rescale 
        if (oldParams.GetTotalProcs() == newParams.totalProcs) {

            // Same grid, do nothing
            if (oldParams.GetNodes() == newParams.GetNodes())
                newMat = new DER(*mat);

            // One to one redistribution
            
            int sendRank = -1;
            int recvRank = -1;

            if (isInOldGrid(rank)) {
                sendRank = newGridToWorldRank[rank];
            }

            if (isInNewGrid(rank)) {
                recvRank = oldGridToWorldRank[rank];
            }
            
            //newMat = SpParHelper::SingleSendSingleRecvMatrix(sendRank, recvRank, mat);

        }


        // Scaling up
        if (oldParams.GetTotalProcs() < newParams.totalProcs)
        {

            int superTileDim = RoundedSqrt<int,int>(newParams.totalProcs) / RoundedSqrt<int,int>(oldParams.totalProcs);
            int superTileSize = superTileDim * superTileDim;

            int recvRankIdx = GetSuperTileIdx(rankInNewGridRow, rankInNewGridCol,
                                                superTileDim, superTileDim);
             
            int recvRank = oldGridToWorldRank[recvRankIdx];

#ifdef DEBUG
            debugPtr->Log("Rank " + std::to_string(rank) +
                    " receiving from new grid rank " + std::to_string(recvRank));
            debugPtr->Log("Rank " + std::to_string(rank) + 
                    " receiving from world rank " + std::to_string(recvRank));
#endif

            std::vector<int> sendRanks;
            if (isInOldGrid(rank))
            {
                for (auto const& elem : newGridToWorldRank)
                {
                    int currNewGridRank = elem.first;
                    int currRankInNewGridRow = currNewGridRank % newGridDim;
                    int currRankInNewGridCol = currNewGridRank / newGridDim;

                    int recvRankInOldGrid = GetSuperTileIdx(currRankInNewGridRow, currRankInNewGridCol,
                                                            superTileDim, superTileDim);
                    int recvRankWorld = oldGridToWorldRank[recvRankInOldGrid];
                    if (recvRankWorld==rank)
                        sendRanks.push_back(elem.second);
                }
#ifdef DEBUG
                debugPtr->LogVecSameLine(sendRanks, "Send Ranks");
#endif
            }

            //newMat = SpParHelper::MultSendSingleRecvMatrix(sendRanks, recvRank, mat);
        }

        // Scaling down
        if (oldParams.GetTotalProcs() > newParams.totalProcs)
        {

            int superTileDim = RoundedSqrt<int,int>(oldParams.totalProcs) / RoundedSqrt<int,int>(newParams.totalProcs);
            int superTileSize = superTileDim * superTileDim;

            // Map each rank to a receiving rank in the new grid

            // Figure out which rank I'm sending to
            // This should be the rank in new grid of the process I'm sending to
            int sendRankIdx = GetSuperTileIdx(rankInOldGridRow, rankInOldGridCol,
                                                superTileDim, superTileDim);
            int sendRank = newGridToWorldRank[sendRankIdx];

#ifdef DEBUG
            debugPtr->Log("Rank " + std::to_string(rank) + " sending to new grid rank " + std::to_string(sendRank));
            debugPtr->Log("Rank " + std::to_string(rank) + " sending to world rank " + std::to_string(sendRank));
#endif

            std::vector<int> recvRanks;
            if (isInNewGrid(rank))
            {
                for (auto const& elem : oldGridToWorldRank)
                {
                    int currOldGridRank = elem.first;
                    int currRankInOldGridRow = currOldGridRank % oldGridDim;
                    int currRankInOldGridCol = currOldGridRank / oldGridDim;

                    int sendRankInOldGrid = GetSuperTileIdx(currRankInOldGridRow, currRankInOldGridCol,
                                                            superTileDim, superTileDim);
                    int sendRankWorld = oldGridToWorldRank[sendRankInOldGrid];
                    if (sendRankWorld==rank)
                        recvRanks.push_back(elem.second);
                }
#ifdef DEBUG
                debugPtr->LogVecSameLine(recvRanks, "Recv Ranks");
#endif
            }

            int rankInSuperTileRow = GetRankInSuperTile(rankInOldGridRow, superTileDim);
            int rankInSuperTileCol = GetRankInSuperTile(rankInOldGridCol, superTileDim);

            /*newMat = SpParHelper::SingleSendMultRecvMatrix(recvRanks, sendRank, mat, 
                                                            rankInSuperTileRow, rankInSuperTileCol,
                                                            oldGridDim, oldGridDim);
                                                            */

        }

        return newMat;

    }

    
    /* Utility functions for getting MPI Communicators across symbolic 3D grid */

    MPI_Comm GridComm() {
        int color = rank / gridSize;
        int key = rank % gridSize;
        MPI_Comm gridComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &gridComm);
        return gridComm;
    }


    MPI_Comm RowComm(MPI_Comm gridComm) {
        int gridRank;
        MPI_Comm_rank(gridComm, &gridRank);
        int color = gridRank / gridDim;
        int key = gridRank % gridDim;
        MPI_Comm rowComm;
        MPI_Comm_split(gridComm, color, key, &rowComm);
        return rowComm;
    }


    MPI_Comm ColComm(MPI_Comm gridComm) {
        int gridRank;
        MPI_Comm_rank(gridComm, &gridRank);
        int color = gridRank % gridDim;
        int key = gridRank / gridDim;
        MPI_Comm colComm;
        MPI_Comm_split(gridComm, color, key, &colComm);
        return colComm;
    }

    MPI_Comm WorldComm() {
        int color = (rank < totalProcs) ? 0 : 1;
        int key = rank;
        MPI_Comm worldComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &worldComm);
        return worldComm;
    }


    inline int GetNodes() const {return nodes;}
    inline int GetPPN() const {return ppn;}
    inline int GetLayers() const {return layers;}
    inline int GetTotalProcs() const {return totalProcs;}
    inline int GetGridSize() const {return gridSize;}
    inline int GetGridDim() const {return gridDim;}

private:
    /* Tunable parameters */
    int nodes;
    int ppn;
    int layers;

    /* Other handy info */
    int totalProcs;
    int gridSize;
    int gridDim;



};

std::ostream& operator<<(std::ostream& os, SpGEMMParams& params) {
    os<<params.OutStr();
    return os;
};

}//autotuning





#endif

