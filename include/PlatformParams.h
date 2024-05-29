
#ifndef PLATFORMPARAMS_H
#define PLATFORMPARAMS_H


#include <exception>


namespace autotuning {

using namespace combblas;

/* 
 * TODO: Topology specific params
 * TODO: intrasocket,intranode alpha/beta
 */

class PlatformParams {
public:
    
    //TODO: Provide utilities for measuring members     

    PlatformParams(){}

    //alpha is us
    //beta is bytes/us
    //all memory things are bytes/us
    PlatformParams(float internodeAlpha, float internodeBeta, float intranodeBeta, 
                    int coresPerNode, int devsPerNode, 
                    long peakFLOPS, float costFLOP, long memBW, float costMem): 
        internodeAlpha(internodeAlpha), internodeBeta(internodeBeta), intranodeBeta(intranodeBeta), 

        coresPerNode(coresPerNode), devsPerNode(devsPerNode),

        peakFLOPS(peakFLOPS), costFLOP(costFLOP), memBW(memBW), costMem(costMem)
    {}
    ~PlatformParams(){}
    
    inline float GetInternodeAlpha() const {return internodeAlpha;} 
    inline float GetInternodeBeta() const {return internodeBeta;} 
    inline float GetIntranodeBeta() const {return intranodeBeta;}
    inline int GetCoresPerNode() const {return coresPerNode;}
    inline int GetDevsPerNode() const {return devsPerNode;}
    
    //TODO: Measure local SpGEMM FLOPS and use that instead
    inline long GetPeakFLOPS() const {return peakFLOPS;}
    inline float GetCostFLOP() const {return costFLOP;}
    inline long GetMemBW() const {return memBW;}
    inline float GetCostMem() const {return costMem;}
    
    float MeasureInternodeAlpha() {throw std::runtime_error("Not implemented");}
    float MeasureInternodeBeta() {throw std::runtime_error("Not implemented");}

    

private:
    
    float internodeAlpha;
    float internodeBeta;
    
    float intranodeBeta;

    int coresPerNode;
    int devsPerNode;
    
    long peakFLOPS; //peak flops on a SINGLE core 
    float costFLOP;
    long memBW;
    float costMem;
    
};

// Values for beta and alpha obtained with osu microbenchmarks
// peak = 3.5Ghz * 2 fmadd * 2 pipelines * 8 flops per vector register
PlatformParams perlmutterParams(3.9, //alpha 
                                23980.54, //internode beta
                                42340.33, //intranode beta
                                128, 4, //cores, gpus
                                (3.5*1e9)*2*2*8, //peak FLOPS
                                5.2e-9, //time for single FLOP in us, measured using simple benchmark 
                                39598, //memBW, measured using STREAM
                                1e-3   //cost for memory movement, right now totally random
                                );
PlatformParams fractusParams(4.2, //alpha
                             15676.06, //internode beta
                             15676.06, //intranode beta -- NOTE: These are rough values
                             32, 2,
                             0.0,
                             5.2e-9,
                             0,
                             0
                             );

}//combblas
#endif
