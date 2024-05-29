

#ifndef MERGE_MODEL_H
#define MERGE_MODEL_H

#include "common.h"

namespace autotuning {

using namespace combblas;

struct MergeInfo {

    //TODO

};

class MergeModel {

public:
    virtual double ComputeTime(MergeInfo * info) {
        INVALID_CALL_ERR();
    }

};


/* Use Sampled Compression Ratio to estimate nnz in output tile */
class MergeModelCompression : public MergeModel {

public:
    MergeModelCompression(MergeInfo * info) {
    }


};


}//combblas




#endif
