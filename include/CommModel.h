


#ifndef COMMMODEL_H
#define COMMMODEL_H


#include "common.h"



namespace autotuning {
using namespace combblas;

/* Dataclass containing information a CommModel needs to compute communication time */
template <typename IT>
struct CommInfo {
    int numMsgs;
    IT numBytes;
};

/* Dataclass containing options that determine CommModel behavior  */
struct CommOpts {
    bool intranode; //If true, use intranode bandwidth
};


/* Generic model for communication time */
template <typename IT>
class CommModel {

public:
    CommModel(){}

    virtual double Time(CommInfo<IT> * info, CommOpts * opts) {INVALID_CALL_ERR();}

    virtual int GetWorld() {INVALID_CALL_ERR();}

};


/* T = alpha + bytes/beta */
template <typename IT>
class PostCommModel : public CommModel<IT> {

public:

    PostCommModel(double alpha, double internodeBeta, double intranodeBeta):
        alpha(alpha), internodeBeta(internodeBeta), intranodeBeta(intranodeBeta) 
    {

    }

    double Time(CommInfo<IT> * info, CommOpts * opts) {
        double beta;
        
        if (opts->intranode)
            beta = intranodeBeta;
        else
            beta = internodeBeta;
        
        return info->numMsgs*alpha + info->numBytes/beta;
    }   

private:
    double alpha; double internodeBeta; double intranodeBeta;
    int coresPerNode;

};




}

#endif
