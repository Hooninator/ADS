
#ifndef COMMON_H
#define COMMON_H




#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cassert>
#include <map>
#include <iomanip>
#include <sstream>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/mtSpGEMM.h"

#define STR(arg) std::to_string(arg)

std::ofstream bcastFile;
std::ofstream SpGEMMFile;





#endif

