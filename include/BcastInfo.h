


#ifndef BCASTINFO_H
#define BCASTINFO_H

#include "common.h"
#include "SpParMatInfo.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"
#include "CommModel.h"

namespace autotuning {

using namespace combblas;

enum BcastAlgorithm {
	BCAST_LINEAR, // Root uses nonblocking sends to send data to all processors
	BCAST_CHAIN, // Pass message along each processor in the communicator
	BCAST_SPLIT_BIN_TREE, // Split message at root of tree, move each half down each half of the tree, leaves swap halves
	BCAST_BIN_TREE, // Standard
	BCAST_BINOMIAL, // Split message and use binomial tree
	BCAST_KNOMIAL, // ditto, but tree with radix k
	BCAST_SCATTER_ALLGATHER, // Reserved for large messages and communicator sizes
	BCAST_NONE // Do nothing
} typedef BcastAlgorithm;


//TODO: This is scuffed. We need to see if it works for larger node counts/other matrices
template <typename IT>
BcastAlgorithm SelectBcastAlgSimple(IT msgSize, int commSize) {
	BcastAlgorithm alg;
    if (msgSize==0 || commSize==0)
        alg=BCAST_NONE;
    else if (msgSize < 49000000)
		alg=BCAST_BIN_TREE;
	else if (msgSize < 190000000)
		alg=BCAST_CHAIN;
	else
		alg=BCAST_KNOMIAL;
	return alg;
}


BcastAlgorithm SelectBcastAlgTree() {return BCAST_BIN_TREE;}


//JB: See https://github.com/open-mpi/ompi/blob/f0261cbef73897133177f17351b80eee6111f1bf/ompi/mca/coll/tuned/coll_tuned_decision_fixed.c#L512
// This is pretty much ripped from this function
template <typename IT>
BcastAlgorithm SelectBcastAlg(IT msgSize, int commSize) {

	//JB: For now, only support decisions for up to communicator size of 16, just to see if this is a decent idea first

	// Do nothing if no bcast
	if (msgSize==0 || commSize==0)
		return BCAST_NONE;

	BcastAlgorithm alg;

	if (commSize < 4) {
		if (msgSize < 32) {
			alg = BCAST_LINEAR; //JB: technically this one should be pipeline but probably does not matter
		} else if (msgSize < 256) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 512) {
			alg = BCAST_LINEAR;
		} else if (msgSize < 1024) {
			alg = BCAST_KNOMIAL;
		} else if (msgSize < 32768) {
			alg = BCAST_LINEAR;
		} else if (msgSize < 131072) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 262144) {
			alg = BCAST_CHAIN;
		} else if (msgSize < 524288) {
			alg = BCAST_LINEAR;
		} else if (msgSize < 1048576) {
			alg = BCAST_BINOMIAL;
		} else {
			alg = BCAST_BIN_TREE;
		}
	} else if (commSize < 8) {
		if (msgSize < 64) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 128) {
			alg = BCAST_BINOMIAL;
		} else if (msgSize < 2048) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 8192) {
			alg = BCAST_BINOMIAL;
		} else if (msgSize < 1048576) {
			alg = BCAST_LINEAR;
		} else {
			alg = BCAST_CHAIN;
		}
	} else if (commSize < 16) {
		if (msgSize < 8) {
			alg = BCAST_KNOMIAL;
		} else if (msgSize < 64) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 4096) {
			alg = BCAST_KNOMIAL;
		} else if (msgSize < 16384) {
			alg = BCAST_BIN_TREE;
		} else if (msgSize < 32768) {
			alg = BCAST_BINOMIAL;
		} else {
			alg = BCAST_LINEAR;
		}
	} else if (commSize < 32) {
		if (msgSize < 4096) {
			alg = BCAST_KNOMIAL;
		} else if (msgSize < 1048576) {
			alg = BCAST_BINOMIAL;
		} else {
			alg = BCAST_SCATTER_ALLGATHER;
		}
	} else {
		throw std::runtime_error("Larger comm sizes not supported yet");
	}

	return alg;
}


template <typename IT>
CommInfo<IT> * MakeBcastCommInfo(const int bcastWorldSize,  const IT msgSize) {

	BcastAlgorithm alg = SelectBcastAlgTree(msgSize, bcastWorldSize);
	//BcastAlgorithm alg = SelectBcastAlgTree();

	CommInfo<IT> * info = new CommInfo<IT>();

	//JB: See https://www.sciencedirect.com/science/article/pii/S0743731522000697
	//TODO: How to determine segsize and radix size

	//JB: will definitely need more precise nnz estimator, otherwise this becomes actively harmful
	switch(alg) {

		case BCAST_LINEAR:
		{
			info->numMsgs = bcastWorldSize;
			info->numBytes = static_cast<IT>(msgSize*bcastWorldSize);
			break;
		}

		case BCAST_CHAIN: //JB: Why would you ever use this one, binomial seems unambiguously superior?
		{
			//Assume split into segments equal to the number of processors/4
			size_t numSegs = bcastWorldSize;
			size_t segSize = msgSize / numSegs;
			info->numMsgs = bcastWorldSize + (numSegs) - 2;
			info->numBytes = segSize * info->numMsgs;
			break;
		}

		case BCAST_SPLIT_BIN_TREE:
		{
			//TODO
			break;
		}

		case BCAST_BIN_TREE:
		{
			//TODO: Paper claims this one is also segment-based
			info->numMsgs = static_cast<int>(log2(bcastWorldSize));
			info->numBytes = msgSize*static_cast<IT>(log2(bcastWorldSize));
			break;
		}

		case BCAST_BINOMIAL:
		{
			//Assume split into segments equal to the number of processors
			size_t numSegs = bcastWorldSize;
			size_t segSize = msgSize / numSegs;
			info->numMsgs = static_cast<int>(log2(bcastWorldSize)) + numSegs - 1;
			info->numBytes = segSize * info->numMsgs;
			break;
		}

		case BCAST_KNOMIAL:
		{
			//Assume split into segments equal to the number of processors
			size_t numSegs = bcastWorldSize;
			size_t segSize = msgSize / numSegs;

			//Assume radix=4 or bcastWorldSize arbitrarily
			size_t radix = 4 > bcastWorldSize ? bcastWorldSize : 4;

			info->numMsgs = (radix - 1) * static_cast<int>(log2(bcastWorldSize) / log2(radix) );
			info->numBytes = segSize * info->numMsgs;

			break;
		}

		case BCAST_SCATTER_ALLGATHER:
		{

			info->numMsgs = static_cast<int>(log2(bcastWorldSize)) + (bcastWorldSize-1);
			info->numBytes = static_cast<IT>(std::lround(msgSize*2*( static_cast<float>(bcastWorldSize - 1) /
																		static_cast<float>(bcastWorldSize) )));
			break;
		}

		default:
		{
			throw std::runtime_error("Bcast algorithm " + std::to_string(alg) + " not supported");
		}

	}

#ifdef DEBUG
	debugPtr->Log("Bcast algorithm: " + std::to_string(alg));
	debugPtr->Log("Msg size: " + std::to_string(msgSize));
#endif

	return info;

}


}//combblas




#endif
