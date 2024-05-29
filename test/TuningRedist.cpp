

#include <cassert>
#include <string>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "Autotuner.h"

#define ATIMING
#define THREADED

using namespace combblas;
using namespace autotuning;

typedef int64_t IT;
typedef double UT;
typedef SpDCCols<IT,UT> DER;
typedef PlusTimesSRing<UT,UT> PTTF;

unsigned int NextSmallestPerfectSquare(unsigned int n) {
    // Find the square root of the input number
    unsigned int root = static_cast<unsigned int>(std::sqrt(n));
    
    // If the square root is not a perfect square, add 1 to it
    if (std::sqrt(root) != std::ceil(std::sqrt(root))) {
        root++;
    }
    
    // Return the square of the next integer
    return root * root;
}


std::string GetMatnameFromPath(const std::string& path)
{
    size_t start = path.rfind('/') + 1; // +1 to start after '/'
    size_t end = path.rfind('.');
    std::string fileName = path.substr(start, end - start);
    return fileName;
}


void RunRedistTest(std::string matpathA, int rank, 
                    SpGEMMParams& startParams, 
                    SpGEMMParams& downParams)
{

    std::string matname = GetMatnameFromPath(matpathA);

    if (rank==0) {
        startParams.Print();
        downParams.Print();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto startGrid = startParams.MakeGridFromParams();
    auto downGrid = downParams.MakeGridFromParams();

    if (startGrid!=NULL) {

        SpParMat<IT,UT,DER> Astart(startGrid);
        Astart.ParallelReadMM(matpathA, true, maximum<double>());
        Astart.ParallelWriteMM(matname+"-correct.mtx", true);

        /**** TEST SCALING DOWN ****/
        /////////////////////////////

        if (rank==0)
            std::cout<<"Testing scaling down..."<<std::endl;

        // Re-distribute from default to smaller process grid
        DER * AdownLocalRedist = downParams.ReDistributeSpMat(Astart.seqptr(), 
                                                                startParams);

        MPI_Barrier(Astart.getcommgrid()->GetWorld());

        if (downGrid!=NULL) 
        {
            DER * AdownLocalRedistCpy = new DER(*AdownLocalRedist);
            SpParMat<IT,UT,DER> AdownRedist(AdownLocalRedistCpy, downGrid);
            AdownRedist.ParallelWriteMM(matname+"-down.mtx", true );
        }


        if (rank==0)
            std::cout<<"Scaling down done"<<std::endl;

        ///////////////////////////
        /**** END TEST SCALING DOWN ****/
       
        MPI_Barrier(Astart.getcommgrid()->GetWorld());

        /**** TEST SCALING UP ****/
        ///////////////////////////
        if (rank==0)
            std::cout<<"Testing scaling up..."<<std::endl;

        // Re-redistribute from smaller to default process grid
        DER * AupLocalRedist = startParams.ReDistributeSpMat(AdownLocalRedist, downParams);


        if (startGrid!=NULL)
        {
            DER * AupLocalRedistCpy = new DER(*AupLocalRedist);
            SpParMat<IT,UT,DER> AupRedist(AupLocalRedistCpy, startGrid);
            AupRedist.ParallelWriteMM(matname+"-up.mtx", true);
        }

        MPI_Barrier(Astart.getcommgrid()->GetWorld());

        if (rank==0)
            std::cout<<"Scaling up done"<<std::endl;

        /**** END TEST SCALING UP ****/
        ///////////////////////////
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
}


void parse_args(int argc, char ** argv,
                std::string& matpath,
                int& startNodes, int& startPPN,
                int& downNodes, int& downPPN)
{
    for (int i=1; i<argc; i++) {

        if (!strcmp(argv[i], "--startnodes"))
            startNodes = std::atoi(argv[i+1]);
        if (!strcmp(argv[i], "--downnodes"))
            downNodes = std::atoi(argv[i+1]);
        if (!strcmp(argv[i], "--startppn"))
            startPPN = std::atoi(argv[i+1]);
        if (!strcmp(argv[i], "--downppn"))
            downPPN = std::atoi(argv[i+1]);
        if (!strcmp(argv[i], "--matpath"))
            matpath = std::string(argv[i+1]);

    }
    
}


int main(int argc, char ** argv) 
{
    
    /* ./<binary> --matpath <path/to/mat> --startnodes <i> --startppn <j>
     * --downnodes <k> --downppn <l> */
    
    assert(argc>10);

    std::string matpathA;
    int startNodes; int downNodes;
    int startPPN; int downPPN;

    parse_args(argc, argv, matpathA, startNodes, startPPN, downNodes, downPPN);
    
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    {
        autotuning::Init(autotuning::M_OMPI);
        autotuning::Autotuner tuner(autotuning::fractusParams);


        SpGEMMParams startParams(startNodes, startPPN, 1);
        SpGEMMParams downParams(downNodes, downPPN, 1);

        RunRedistTest(matpathA, rank, startParams, downParams);

    }

    MPI_Finalize();

}
