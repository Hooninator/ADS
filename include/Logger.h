
#ifndef LOGGER_H
#define LOGGER_H


#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

namespace autotuning {
using namespace combblas;

class Logger {
public:
    
    Logger(int rank, std::string name, bool allRanks): rank(rank), _allRanks(allRanks) {
        ofs.open(name, std::ofstream::out);
    }

    
    template <typename T>
    void Print(T msg) {
        if (rank==0 ||_allRanks ) std::cout<<msg<<std::endl;
    }
    
    
    template <typename T>
    void Print0(T msg) {
        if (rank==0) std::cout<<msg<<std::endl;
    }

    
    template <typename T>
    void Log(T msg) {
        if (rank==0 || _allRanks) ofs<<msg<<std::endl;
    }


    template <typename T>
    void LogVec(std::vector<T>& v) {
        if (rank==0||_allRanks) {
            std::for_each(v.begin(), v.end(), [this](T& elem) {this->ofs<<elem<<std::endl;});
        }
    }

    template <typename T>
    void LogVecSameLine(std::vector<T>& v, std::string prefix="") {
        if (rank==0||_allRanks) {
            this->ofs<<prefix<<":";
            std::for_each(v.begin(), v.end(), [this](T& elem) {this->ofs<<elem<<" ";});
            this->ofs<<std::endl;
        }
    }

    ~Logger(){ofs.close();}

private:
    int rank;
    bool _allRanks;
    std::ofstream ofs;

};

}//combblas

#endif

