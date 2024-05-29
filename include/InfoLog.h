#ifndef INFOLOG_H
#define INFOLOG_H

#include <fstream>
#include <map>

namespace autotuning {
using namespace combblas;

class InfoLog {
public:
    InfoLog(const std::string fileName, int rank) : rank(rank) {
        ofs.open(fileName, std::ofstream::out);
    }


    /* Timer utilities */
    void StartTimer(const std::string label) { 
        double stime = MPI_Wtime(); 
        infoMap[label]= std::to_string(stime);
    }

    void EndTimer(const std::string label) {
        double etime = MPI_Wtime();
        double totalTime = etime - std::stod(infoMap[label]);
        infoMap[label] = std::to_string(totalTime);
    }

    void StartTimerGlobal(const std::string label) { 
        double stime = MPI_Wtime(); 
        infoMapGlobal[label] = std::to_string(stime);
    }

    void EndTimerGlobal(const std::string label) {
        double etime = MPI_Wtime();
        double totalTime = etime - std::stod(infoMapGlobal[label]);
        infoMapGlobal[label] = std::to_string(totalTime);
    }
    

    /* Write all info into file */
    void WriteInfo() {
        ofs<<"----PREDICTION INFO----"<<std::endl;
        std::for_each(infoMap.begin(), infoMap.end(),
            [this](auto const& elem) {
                this->ofs<<elem.first<<":"<<elem.second<<" ";
            }
        );
        ofs<<std::endl;
    }

    void WriteInfoGlobal() {

        ofs<<std::endl;

        ofs<<"----GLOBALS RANK " <<rank<<"----"<<std::endl;
        std::for_each(infoMapGlobal.begin(), infoMapGlobal.end(),
            [this](auto const& elem) {
                this->ofs<<elem.first<<":"<<elem.second<<std::endl;
            }
        );
        ofs<<std::endl;
        
        float sum = std::atof(infoMapGlobal[std::string("TuneSpGEMM2DPhase")].c_str());

        //TODO: better skipping
        ofs<<"----PERCENTAGES RANK " <<rank<<"----"<<std::endl;
        std::for_each(infoMapGlobal.begin(), infoMapGlobal.end(),
            [this, sum](auto const& elem) {
                if (!strcmp(elem.first.c_str(), "FeatureInit") || !strcmp(elem.first.c_str(), "Prediction")) {
                    this->ofs<<elem.first<<":"<<std::atof(elem.second.c_str())*100/sum<<"% "<<std::endl;
                }
            }
        );
        ofs<<std::endl;

    }

    void WriteOne(const std::string key) {
        this->ofs<<key<<":"<<infoMap[key]<<std::endl;
    }

    void WriteOneGlobal(const std::string key) {
        this->ofs<<key<<":"<<infoMapGlobal[key]<<std::endl;
    }


    /* Print somerhing */
    void Print(const std::string label) {
        if (rank==0) {
            std::cout<<"["<<label<<"]"<<": "<<infoMap[label]<<std::endl;
        }
    }

    void PrintGlobal(const std::string label) {
        if (rank==0) {
            std::cout<<"["<<label<<"]"<<": "<<infoMapGlobal[label]<<std::endl;
        }
    }


    /* Put and get utilities */
    std::string Get(const std::string label) {
        return infoMap[label];
    }

    template <typename T>
    void Put(const std::string key, const T value) {
        infoMap[key] = std::to_string(value);
    }

    void Put(const std::string key, const std::string value) {
        infoMap[key] = value; 
    }

    std::string GetGlobal(const std::string label) {
        return infoMapGlobal[label];
    }

    template <typename T>
    void PutGlobal(const std::string key, const T value) {
        infoMapGlobal[key] = std::to_string(value);
    }

    void PutGlobal(const std::string key, const std::string value) {
        infoMapGlobal[key] = value;
    }


    /* Clear maps */
    void Clear() {infoMap.clear();}
    void ClearGlobal() {infoMapGlobal.clear();}

    std::ofstream& OFS(){return ofs;}//TODO: This is horrible

    std::map<std::string, std::string> GetInfoMap() {return infoMap;}
    std::map<std::string, std::string> GetInfoMapGlobal() {return infoMapGlobal;}

    ~InfoLog() {ofs.close();}


private:
    std::ofstream ofs;
    std::map<std::string, std::string> infoMap;
    std::map<std::string, std::string> infoMapGlobal;
    int rank;

};

}




#endif




