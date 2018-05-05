//
// Created by 范宏昌 on 2018/4/13.
//

#include "Saver.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include <cstdio>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <cstring>

#define SAVER_DEBUG
#define WRITE(ele, file) fwrite(&(ele),sizeof(ele),1,file)
#define READ(ele, file) do{ \
        if(fread(&ele,sizeof(ele),1,file)==0) \
        {printf("\nEOF\n");} \
        }while(0)
#define WRITEMAT(D, file) do{ \
        int type=D.type();\
        int sizeImg[2] = { (D).cols , (D).rows }; \
        fwrite(&type,1,sizeof(int),file); \
        fwrite(sizeImg, 2, sizeof(int), file); \
        fwrite((D).data, (D).cols * (D).rows, (D).elemSize(), file); \
    }while(0)

#define READMAT(D, file) do { \
        int sizeImg[2];int type; \
        if(fread(&type,1,sizeof(int),file)==0){printf("\nEOF\n");} \
        if(fread(sizeImg, 2, sizeof(int), file)==0){printf("\nEOF\n");} \
        (D).create(sizeImg[1], sizeImg[0],type); \
        if(fread((D).data,sizeImg[1] * sizeImg[0], (D).elemSize(),file)==0) \
        {printf("\nEOF\n");} \
    } while(0)

namespace ORB_SLAM2 {


Saver::Saver(string SLAMsrcfile) : filename(SLAMsrcfile)
{}

void Saver::setclasses(KeyFrameDatabase *pKeyFrameDatabase, Map *pmap, ORBVocabulary *vocabulary)
{
    KB = pKeyFrameDatabase;
    map = pmap;
    mpORBvocabulary = vocabulary;
}

void Saver::loadOneKeyframe(FILE *file)
{
    long unsigned int mnId;
    long unsigned int mnFrameId;
    
    double mTimeStamp;
    
    // Grid (to speed up feature matching)
    int mnGridCols;
    int mnGridRows;
    float mfGridElementWidthInv;
    float mfGridElementHeightInv;
    
    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;
    
    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;
    
    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;
    
    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;
    
    // Calibration parameters
    float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
    
    // Number of KeyPoints
    int N;
    
    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<float> mvuRight; // negative value for monocular points
    std::vector<float> mvDepth; // negative value for monocular points
    cv::Mat mDescriptors;
    
    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    
    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;
    
    // Scale
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
    
    // Image bounds and calibration
    int mnMinX;
    int mnMinY;
    int mnMaxX;
    int mnMaxY;
    cv::Mat mK;
    
    
    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;
    
    cv::Mat Cw; // Stereo middel point. Only for visualization
    
    // MapPoints associated to keypoints
//    std::vector<MapPoint *> mvpMapPoints;
    
    // BoW
    ORBVocabulary *mpORBvocabulary;
    
    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t> > > mGrid;

//    std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
//    std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    
    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
//    std::set<KeyFrame *> mspChildrens;
//    std::set<KeyFrame *> mspLoopEdges;
    
    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;
    
    float mHalfBaseline; // Only for visualization
    
    
    READ(mnId, file);
    READ(mnFrameId, file);
    READ(mTimeStamp, file);
    READ(mnGridCols, file);
    READ(mnGridRows, file);
    READ(mfGridElementWidthInv, file);
    READ(mfGridElementHeightInv, file);
    READ(mnTrackReferenceForFrame, file);
    READ(mnFuseTargetForKF, file);
    READ(mnBALocalForKF, file);
    READ(mnBAFixedForKF, file);
    READ(mnLoopQuery, file);
    READ(mnLoopWords, file);
    READ(mLoopScore, file);
    READ(mnRelocQuery, file);
    READ(mnRelocWords, file);
    READ(mRelocScore, file);
    READMAT(mTcwGBA, file);
    READMAT(mTcwBefGBA, file);
    READ(mnBAGlobalForKF, file);
    READ(fx, file);
    READ(fy, file);
    READ(cx, file);
    READ(cy, file);
    READ(invfx, file);
    READ(invfy, file);
    READ(mbf, file);
    READ(mb, file);
    READ(mThDepth, file);
    READ(N, file);

//    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
//    const std::vector<cv::KeyPoint> mvKeys;
//    const std::vector<cv::KeyPoint> mvKeysUn;
//    const std::vector<float> mvuRight; // negative value for monocular points
//    const std::vector<float> mvDepth; // negative value for monocular points
//    const cv::Mat mDescriptors;
    unsigned long size;
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        cv::KeyPoint tmp;
        READ(tmp, file);
        mvKeys.push_back(tmp);
    }
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        cv::KeyPoint tmp;
        READ(tmp, file);
        mvKeysUn.push_back(tmp);
    }
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        float tmp;
        READ(tmp, file);
        mvuRight.push_back(tmp);
    }
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        float tmp;
        READ(tmp, file);
        mvDepth.push_back(tmp);
    }
    READMAT(mDescriptors, file);

//    //BoW
//    DBoW2::BowVector mBowVec;
//    DBoW2::FeatureVector mFeatVec;
    READMAT(mTcp, file);
    READ(mnScaleLevels, file);
    READ(mfScaleFactor, file);
    READ(mfLogScaleFactor, file);
    
    
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        float tmp;
        READ(tmp, file);
        mvScaleFactors.push_back(tmp);
    }
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        float tmp;
        READ(tmp, file);
        mvLevelSigma2.push_back(tmp);
    }
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        float tmp;
        READ(tmp, file);
        mvInvLevelSigma2.push_back(tmp);
    }
    
    READ(mnMinX, file);
    READ(mnMinY, file);
    READ(mnMaxX, file);
    READ(mnMaxY, file);
    READMAT(mK, file);
    
    READMAT(Tcw, file);
    READMAT(Twc, file);
    READMAT(Ow, file);
    READMAT(Cw, file);
    
    // TODO ID : need replace ----
    std::vector<long unsigned int> mvpMapPoints_id;
    READ(size, file);
    for (unsigned long i = 0; i < size; i++) {
        long unsigned int tmp;
        READ(tmp, file);
        mvpMapPoints_id.push_back(tmp);
    }
    mvpMapPoints_id_all.push_back(mvpMapPoints_id);
    
    READ(size, file);
    for (unsigned long i = 0; i < size; ++i) {
        unsigned long size2;
        std::vector<std::vector<size_t>> tmp1;
        READ(size2, file);
        for (unsigned long j = 0; j < size2; ++j) {
            unsigned long size3;
            std::vector<size_t> tmp2;
            READ(size3, file);
            for (unsigned long k = 0; k < size3; ++k) {
                size_t tmp3;
                READ(tmp3, file);
                tmp2.push_back(tmp3);
            }
            tmp1.push_back(tmp2);
        }
        mGrid.push_back(tmp1);
    }

// TODO : need replace-----
//    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    READ(size, file);
    std::vector<long unsigned int> mConnectedKeyFrameWeights_id;
    std::vector<int> mConnectedKeyFrameWeights_weight;
    for (unsigned long i = 0; i < size; ++i) {
        long unsigned int tmp1;
        int tmp2;
        READ(tmp1, file);
        READ(tmp2, file);
        mConnectedKeyFrameWeights_id.push_back(tmp1);
        mConnectedKeyFrameWeights_weight.push_back(tmp2);
    }
    mConnectedKeyFrameWeights_id_all.push_back(mConnectedKeyFrameWeights_id);
    mConnectedKeyFrameWeights_weight_all.push_back(mConnectedKeyFrameWeights_weight);
    
    //TODO : need replace----
//    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    READ(size, file);
    std::vector<long unsigned int> mvpOrderedConnectedKeyFrames_id;
    for (unsigned long i = 0; i < size; ++i) {
        long unsigned int tmp;
        READ(tmp, file);
        mvpOrderedConnectedKeyFrames_id.push_back(tmp);
    }
    mvpOrderedConnectedKeyFrames_id_all.push_back(mvpOrderedConnectedKeyFrames_id);


//    std::vector<int> mvOrderedWeights;
    READ(size, file);
    for (unsigned long i = 0; i < size; ++i) {
        int tmp;
        READ(tmp, file);
        mvOrderedWeights.push_back(tmp);
    }
    
    
    
    // Spanning Tree and Loop Edges
//    bool mbFirstConnection;
    READ(mbFirstConnection, file);
    //TODO : need replace----
//    KeyFrame* mpParent;
    long unsigned int mpParent_id;
    READ(mpParent_id, file);
    mpParent_id_all.push_back(mpParent_id);
    //TODO need replace----
//    std::set<KeyFrame*> mspChildrens;
    READ(size, file);
    std::vector<long unsigned int> mspChildrens_id;
    for (unsigned long i = 0; i < size; ++i) {
        long unsigned int tmp;
        READ(tmp, file);
        mspChildrens_id.push_back(tmp);
    }
    mspChildrens_id_all.push_back(mspChildrens_id);
    
    //TODO need replace----
//    std::set<KeyFrame*> mspLoopEdges;
    std::vector<long unsigned int> mspLoopEdges_id;
    READ(size, file);
    for (unsigned long i = 0; i < size; ++i) {
        long unsigned int tmp;
        READ(tmp, file);
        mspLoopEdges_id.push_back(tmp);
    }
    mspLoopEdges_id_all.push_back(mspLoopEdges_id);

//    // Bad flags
//    bool mbNotErase;
//    bool mbToBeErased;
//    bool mbBad;
    READ(mbNotErase, file);
    READ(mbToBeErased, file);
    READ(mbBad, file);
    READ(mHalfBaseline, file);

#ifdef SAVER_DEBUG
    char check[8];
    fread(check,1,8,file);
    if(strcmp(check,"kfcheck")!=0){
        printf("Saver error in kfid: %d\n",mnId);
    }
#endif

//    Map* mpMap;
    KeyFrame *tmp;
    tmp = new KeyFrame(mnId, mnFrameId, mTimeStamp, mfGridElementWidthInv, mfGridElementHeightInv, fx, fy, cx, cy,
                       invfx,
                       invfy,
                       mbf, mb, mThDepth, N, mvKeys, mvKeysUn, mvuRight, mvDepth, mDescriptors, mnScaleLevels,
                       mfScaleFactor,
                       mfLogScaleFactor, mvScaleFactors, mvLevelSigma2, mvInvLevelSigma2, mnMinX, mnMinY, mnMaxX,
                       mnMaxY, mK,
                       map,
                       KB);
    tmp->mnTrackReferenceForFrame = mnTrackReferenceForFrame;
    tmp->mnFuseTargetForKF = mnFuseTargetForKF;
    tmp->mnBALocalForKF = mnBALocalForKF;
    tmp->mnBAFixedForKF = mnBAFixedForKF;
    tmp->mnLoopQuery = mnLoopQuery;
    
    tmp->mnLoopQuery = mnLoopQuery;
    tmp->mnLoopWords = mnLoopWords;
    tmp->mLoopScore = mLoopScore;
    tmp->mnRelocQuery = mnRelocQuery;
    tmp->mnRelocWords = mnRelocWords;
    tmp->mRelocScore = mRelocScore;
    
    tmp->mTcwGBA = mTcwGBA;
    tmp->mTcwBefGBA = mTcwBefGBA;
    tmp->mnBAGlobalForKF = mnBAGlobalForKF;
    
    tmp->ComputeBoW();
    
    tmp->mTcp = mTcp;
    tmp->Tcw = Tcw;
    tmp->Twc = Twc;
    tmp->Ow = Ow;
    tmp->Cw = Cw;
    tmp->mpORBvocabulary = mpORBvocabulary;
    
    tmp->mGrid = mGrid;
    
    tmp->mvOrderedWeights = mvOrderedWeights;
    
    // Spanning Tree and Loop Edges
    tmp->mbFirstConnection = mbFirstConnection;
    
    tmp->mbNotErase = mbNotErase;
    tmp->mbToBeErased = mbToBeErased;
    tmp->mbBad = mbBad;
    tmp->mHalfBaseline = mHalfBaseline;
    
    kfv.push_back(tmp);
    kfmap.insert(std::pair<long unsigned int, KeyFrame *>(tmp->mnId, tmp));
}

void Saver::loadOneMapPoint(FILE *file)
{
    long unsigned int mnId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;
    
    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;
    
    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;
    
    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;
    
    
    static std::mutex mGlobalMutex;
    
    
    // Position in absolute coordinates
    cv::Mat mWorldPos;
    
    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;
    
    // Mean viewing direction
    cv::Mat mNormalVector;
    
    // Best descriptor to fast matching
    cv::Mat mDescriptor;
    
    
    // Tracking counters
    int mnVisible;
    int mnFound;
    
    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    
    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;
    
    
    unsigned long size;
//    long unsigned int mnId;
    READ(mnId, file);
//    long int mnFirstKFid;
    READ(mnFirstKFid, file);
//    long int mnFirstFrame;
    READ(mnFirstFrame, file);
//    int nObs;
    READ(nObs, file);
//
//    // Variables used by the tracking
//    float mTrackProjX;
//    float mTrackProjY;
//    float mTrackProjXR;
//    bool mbTrackInView;
//    int mnTrackScaleLevel;
//    float mTrackViewCos;
//    long unsigned int mnTrackReferenceForFrame;
//    long unsigned int mnLastFrameSeen;
    READ(mTrackProjX, file);
    READ(mTrackProjY, file);
    READ(mTrackProjXR, file);
    READ(mbTrackInView, file);
    READ(mnTrackScaleLevel, file);
    READ(mTrackViewCos, file);
    READ(mnTrackReferenceForFrame, file);
    READ(mnLastFrameSeen, file);
//
//    // Variables used by local mapping
//    long unsigned int mnBALocalForKF;
//    long unsigned int mnFuseCandidateForKF;
//
//    // Variables used by loop closing
//    long unsigned int mnLoopPointForKF;
//    long unsigned int mnCorrectedByKF;
//    long unsigned int mnCorrectedReference;
    READ(mnBALocalForKF, file);
    READ(mnFuseCandidateForKF, file);
    READ(mnLoopPointForKF, file);
    READ(mnCorrectedByKF, file);
    READ(mnCorrectedReference, file);

//    cv::Mat mPosGBA;
    READMAT(mPosGBA, file);
//    long unsigned int mnBAGlobalForKF;
    READ(mnBAGlobalForKF, file);

//    // Position in absolute coordinates
//    cv::Mat mWorldPos;
    READMAT(mWorldPos, file);
//
    // TODO : need replace-----
//    // Keyframes observing the point and associated index in keyframe
//    std::map<KeyFrame *, size_t> mObservations;
    READ(size, file);
    std::vector<long unsigned int> mObservations_id;
    std::vector<size_t> mObservations_size_t;
    for (unsigned long i = 0; i < size; i++) {
        long unsigned int tmp1;
        size_t tmp2;
        READ(tmp1, file);
        READ(tmp2, file);
        mObservations_id.push_back(tmp1);
        mObservations_size_t.push_back(tmp2);
    }
    mObservations_id_all.push_back(mObservations_id);
    mObservations_size_t_all.push_back(mObservations_size_t);

//    // Mean viewing direction
//    cv::Mat mNormalVector;
    READMAT(mNormalVector, file);
//
//    // Best descriptor to fast matching
//    cv::Mat mDescriptor;
    READMAT(mDescriptor, file);
//
    //TODO : need replace----
//    // Reference KeyFrame
//    KeyFrame *mpRefKF;
    long unsigned int mpRefKF_id;
    READ(mpRefKF_id, file);//ID
    mpRefKF_id_all.push_back(mpRefKF_id);

//
//    // Tracking counters
//    int mnVisible;
//    int mnFound;
    READ(mnVisible, file);
    READ(mnFound, file);

//
//    // Bad flag (we do not currently erase MapPoint from memory)
//    bool mbBad;
    READ(mbBad, file);
    
    //TODO :need replace----
//    MapPoint *mpReplaced;
    long unsigned int mpReplaced_id;
    READ(mpReplaced_id, file);
    mpReplaced_id_all.push_back(mpReplaced_id);

//
//    // Scale invariance distances
//    float mfMinDistance;
//    float mfMaxDistance;
    READ(mfMinDistance, file);
    READ(mfMaxDistance, file);
//
//    Map *mpMap;
#ifdef SAVER_DEBUG
    char check[8];
    fread(check,1,8,file);
    if(strcmp(check,"mpcheck")!=0){
        printf("Saver error in mpid: %d\n",mnId);
    }
#endif

    
    MapPoint *mappoint;
    mappoint = new MapPoint(map);
    
    mappoint->mnId = mnId;
    mappoint->mnFirstKFid = mnFirstKFid;
    mappoint->mnFirstFrame = mnFirstFrame;
    mappoint->nObs = nObs;
    
    mappoint->mTrackProjX = mTrackProjX;
    mappoint->mTrackProjY = mTrackProjY;
    mappoint->mTrackProjXR = mTrackProjXR;
    mappoint->mbTrackInView = mbTrackInView;
    mappoint->mnTrackScaleLevel = mnTrackScaleLevel;
    mappoint->mTrackViewCos = mTrackViewCos;
    mappoint->mnTrackReferenceForFrame = mnTrackReferenceForFrame;
    mappoint->mnLastFrameSeen = mnLastFrameSeen;
    
    mappoint->mnBALocalForKF = mnBALocalForKF;
    mappoint->mnFuseCandidateForKF = mnFuseCandidateForKF;
    mappoint->mnLoopPointForKF = mnLoopPointForKF;
    mappoint->mnCorrectedByKF = mnCorrectedByKF;
    mappoint->mnCorrectedReference = mnCorrectedReference;
    mappoint->mPosGBA = mPosGBA;
    mappoint->mnBAGlobalForKF = mnBAGlobalForKF;
    
    mappoint->mWorldPos = mWorldPos;
    mappoint->mNormalVector = mNormalVector;
    mappoint->mDescriptor = mDescriptor;
    
    mappoint->mnVisible = mnVisible;
    mappoint->mnFound = mnFound;
    
    mappoint->mbBad = mbBad;
    
    mappoint->mfMinDistance = mfMinDistance;
    mappoint->mfMaxDistance = mfMaxDistance;
    
    mpv.push_back(mappoint);
    mpmap.insert(std::pair<long unsigned int, MapPoint *>(mappoint->mnId, mappoint));
}

void Saver::replaceAllIndex()
{
    /*
     * keyframes
     */

    unsigned long size = mvpMapPoints_id_all.size();
    for (unsigned long t = 0; t < size; t++) {
        unsigned long size2;
        size2 = mvpMapPoints_id_all[t].size();
        for (unsigned long i = 0; i < size2; i++) {
            kfv[t]->mvpMapPoints.push_back((*mpmap.find(mvpMapPoints_id_all[t][i])).second);
        }
        
        //    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
        size2 = mConnectedKeyFrameWeights_id_all.size();
        for (unsigned long i = 0; i < size2; ++i) {
            kfv[t]->mConnectedKeyFrameWeights.insert(
                std::pair<KeyFrame *, int>((*kfmap.find(mConnectedKeyFrameWeights_id_all[t][i])).second,
                                           mConnectedKeyFrameWeights_weight_all[t][i]));
        }
        
        //    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
        size2 = mvpOrderedConnectedKeyFrames_id_all.size();
        for (unsigned long i = 0; i < size2; ++i) {
            kfv[t]->mvpOrderedConnectedKeyFrames.push_back(
                (*kfmap.find(mvpOrderedConnectedKeyFrames_id_all[t][i])).second);
        }
        
        //    KeyFrame* mpParent;
        kfv[t]->mpParent = (*kfmap.find(mpParent_id_all[t])).second;
        
        //    std::set<KeyFrame*> mspChildrens;
        size2 = mspChildrens_id_all.size();
        for (unsigned long i = 0; i < size2; ++i) {
            kfv[t]->mspChildrens.insert((*kfmap.find(mspChildrens_id_all[t][i])).second);
        }
        
        //    std::set<KeyFrame*> mspLoopEdges;
        size2 = mspLoopEdges_id_all.size();
        for (unsigned long i = 0; i < size2; ++i) {
            kfv[t]->mspLoopEdges.insert((*kfmap.find(mspLoopEdges_id_all[t][i])).second);
        }
        
        
    }
    
    /*
     * mappoints
     */
    size = mObservations_id_all.size();
    
    
    //    // Keyframes observing the point and associated index in keyframe
//    std::map<KeyFrame *, size_t> mObservations;
    for (unsigned long t = 0; t < size; t++) {
        unsigned long size2=mObservations_id_all[t].size();
        
        for (unsigned long i = 0; i < size2; i++) {
            mpv[t]->mObservations.insert(std::pair<KeyFrame *, size_t>((*kfmap.find(mObservations_id_all[t][i])).second,mObservations_size_t_all[t][i]));
        }
        
        //    KeyFrame *mpRefKF;
        mpv[t]->mpRefKF=((*kfmap.find(mpRefKF_id_all[t])).second);
    
        //    MapPoint *mpReplaced;
        mpv[t]->mpReplaced=((*mpmap.find(mpReplaced_id_all[t])).second);
    }

}

void Saver::loadfromfile()
{
    FILE *file;
    file = fopen(filename.c_str(), "r");
    if (file == NULL) {
        printf("no saved file found, skip loading.\n");
        return;
    }else{
        printf("Saved file found, starting restore...\n");
    }
    /*
     * Map
     */
    
    // vars in map
    READ(map->mnMaxKFid, file);
    READ(KeyFrame::nNextId, file);
    READ(MapPoint::nNextId, file);
    
    
    // map's KeyFrames
    unsigned long kfsize;
    READ(kfsize, file);
    for (unsigned long i = 0; i < kfsize; ++i) {
        loadOneKeyframe(file);
    }
    
    // map's MapPoints
    unsigned long mpsize;
    READ(mpsize, file);
    for (unsigned long i = 0; i < mpsize; ++i) {
        loadOneMapPoint(file);
    }
    
    // map's mvpKeyFrameOrigins
    // ONLY ID
    unsigned long kfosize;
    READ(kfosize, file);
    for (unsigned long i = 0; i < kfosize; i++) {
//        unsigned long id = map->mvpKeyFrameOrigins[i]->mnId;
        long unsigned int id;
        READ(id, file);
        map->mvpKeyFrameOrigins.push_back((*kfmap.find(id)).second);
    }
    
    // map's mvpReferenceMapPoints
    // ONLY ID
    unsigned long rmpsize;
    READ(rmpsize, file);
    for (unsigned long i = 0; i < rmpsize; i++) {
        long unsigned int id;
        READ(id, file);
        map->mvpReferenceMapPoints.push_back((*mpmap.find(id)).second);
    }
    
    /*
     * KFDatabase
     */
    unsigned long vsize;
    READ(vsize, file);
    for (unsigned long i = 0; i < vsize; i++) {
        unsigned long lsize;
        READ(lsize, file);
        std::list<KeyFrame *> l;
        for (unsigned long j = 0; j < lsize; j++) {
            long unsigned int id;
            READ(id, file);
            l.push_back((*kfmap.find(id)).second);
        }
        KB->mvInvertedFile.push_back(l);
    }
    
    fclose(file);
    printf("Map restored, starting reindexing...\n");
    replaceAllIndex();
    printf("Reindexing finished successfully.\n");
    
}

inline void Saver::saveOneKeyframe(KeyFrame *kf, FILE *file)
{
    unsigned long size;
    
    WRITE(kf->mnId, file);
    WRITE(kf->mnFrameId, file);
    WRITE(kf->mTimeStamp, file);
    WRITE(kf->mnGridCols, file);
    WRITE(kf->mnGridRows, file);
    WRITE(kf->mfGridElementWidthInv, file);
    WRITE(kf->mfGridElementHeightInv, file);
    WRITE(kf->mnTrackReferenceForFrame, file);
    WRITE(kf->mnFuseTargetForKF, file);
    WRITE(kf->mnBALocalForKF, file);
    WRITE(kf->mnBAFixedForKF, file);
    WRITE(kf->mnLoopQuery, file);
    WRITE(kf->mnLoopWords, file);
    WRITE(kf->mLoopScore, file);
    WRITE(kf->mnRelocQuery, file);
    WRITE(kf->mnRelocWords, file);
    WRITE(kf->mRelocScore, file);
    WRITEMAT(kf->mTcwGBA, file);
    WRITEMAT(kf->mTcwBefGBA, file);
    WRITE(kf->mnBAGlobalForKF, file);
    WRITE(kf->fx, file);
    WRITE(kf->fy, file);
    WRITE(kf->cx, file);
    WRITE(kf->cy, file);
    WRITE(kf->invfx, file);
    WRITE(kf->invfy, file);
    WRITE(kf->mbf, file);
    WRITE(kf->mb, file);
    WRITE(kf->mThDepth, file);
    WRITE(kf->N, file);

//    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
//    const std::vector<cv::KeyPoint> mvKeys;
//    const std::vector<cv::KeyPoint> mvKeysUn;
//    const std::vector<float> mvuRight; // negative value for monocular points
//    const std::vector<float> mvDepth; // negative value for monocular points
//    const cv::Mat mDescriptors;
    size = kf->mvKeys.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvKeys.at(i), file);
    }
    size = kf->mvKeysUn.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvKeysUn.at(i), file);
    }
    size = kf->mvuRight.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvuRight.at(i), file);
    }
    size = kf->mvDepth.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvDepth.at(i), file);
    }
    
    WRITEMAT(kf->mDescriptors, file);

//
//    //BoW
//    DBoW2::BowVector mBowVec;
//    DBoW2::FeatureVector mFeatVec;
    WRITEMAT(kf->mTcp, file);
    WRITE(kf->mnScaleLevels, file);
    WRITE(kf->mfScaleFactor, file);
    WRITE(kf->mfLogScaleFactor, file);
    
    size = kf->mvScaleFactors.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvScaleFactors.at(i), file);
    }
    
    size = kf->mvLevelSigma2.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvLevelSigma2.at(i), file);
    }
    
    size = kf->mvInvLevelSigma2.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvInvLevelSigma2.at(i), file);
    }
    
    WRITE(kf->mnMinX, file);
    WRITE(kf->mnMinY, file);
    WRITE(kf->mnMaxX, file);
    WRITE(kf->mnMaxY, file);
    WRITEMAT(kf->mK, file);
    
    WRITEMAT(kf->Tcw, file);
    WRITEMAT(kf->Twc, file);
    WRITEMAT(kf->Ow, file);
    WRITEMAT(kf->Cw, file);
    
    // Uses Index here
    size = kf->mvpMapPoints.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; i++) {
        WRITE(kf->mvpMapPoints.at(i)->mnId, file);
    }


//    // BoW
//    KeyFrameDatabase* mpKeyFrameDB;
//    ORBVocabulary* mpORBvocabulary;
    size = kf->mGrid.size();
    WRITE(size, file);
    for (unsigned long i = 0; i < size; ++i) {
        unsigned long size2 = kf->mGrid[i].size();
        WRITE(size2, file);
        for (unsigned long j = 0; j < size2; ++j) {
            unsigned long size3 = kf->mGrid[i][j].size();
            WRITE(size3, file);
            for (unsigned long k = 0; k < size3; ++k) {
                WRITE(kf->mGrid[i][j][k], file);
            }
        }
    }


//    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    size = kf->mConnectedKeyFrameWeights.size();
    WRITE(size, file);
    for (auto &it:kf->mConnectedKeyFrameWeights) {
        WRITE(it.first->mnId, file);
        WRITE(it.second, file);
    }


//    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    size = kf->mvpOrderedConnectedKeyFrames.size();
    WRITE(size, file);
    for (auto &it:kf->mvpOrderedConnectedKeyFrames) {
        WRITE(it->mnId, file);
    }

//    std::vector<int> mvOrderedWeights;
    size = kf->mvOrderedWeights.size();
    WRITE(size, file);
    for (auto &it:kf->mvOrderedWeights) {
        WRITE(it, file);
    }
    
    
    
    // Spanning Tree and Loop Edges
//    bool mbFirstConnection;
    WRITE(kf->mbFirstConnection, file);
//    KeyFrame* mpParent;
    WRITE(kf->mpParent->mnId, file);
//    std::set<KeyFrame*> mspChildrens;
    size = kf->mspChildrens.size();
    WRITE(size, file);
    for (auto &it:kf->mspChildrens) {
        WRITE(it->mnId, file);
    }

//    std::set<KeyFrame*> mspLoopEdges;
    size = kf->mspLoopEdges.size();
    WRITE(size, file);
    for (auto &it:kf->mspLoopEdges) {
        WRITE(it->mnId, file);
    }

//    // Bad flags
//    bool mbNotErase;
//    bool mbToBeErased;
//    bool mbBad;
    WRITE(kf->mbNotErase, file);
    WRITE(kf->mbToBeErased, file);
    WRITE(kf->mbBad, file);
    WRITE(kf->mHalfBaseline, file);

#ifdef SAVER_DEBUG
    char* check="kfcheck";
    fwrite(check,1,8,file);
#endif
//    Map* mpMap;
}

inline void Saver::saveOneMapPoint(MapPoint *mp, FILE *file)
{
    unsigned long size;
//    long unsigned int mnId;
    WRITE(mp->mnId, file);
//    long int mnFirstKFid;
    WRITE(mp->mnFirstKFid, file);
//    long int mnFirstFrame;
    WRITE(mp->mnFirstFrame, file);
//    int nObs;
    WRITE(mp->nObs, file);
//
//    // Variables used by the tracking
//    float mTrackProjX;
//    float mTrackProjY;
//    float mTrackProjXR;
//    bool mbTrackInView;
//    int mnTrackScaleLevel;
//    float mTrackViewCos;
//    long unsigned int mnTrackReferenceForFrame;
//    long unsigned int mnLastFrameSeen;
    WRITE(mp->mTrackProjX, file);
    WRITE(mp->mTrackProjY, file);
    WRITE(mp->mTrackProjXR, file);
    WRITE(mp->mbTrackInView, file);
    WRITE(mp->mnTrackScaleLevel, file);
    WRITE(mp->mTrackViewCos, file);
    WRITE(mp->mnTrackReferenceForFrame, file);
    WRITE(mp->mnLastFrameSeen, file);
//
//    // Variables used by local mapping
//    long unsigned int mnBALocalForKF;
//    long unsigned int mnFuseCandidateForKF;
//
//    // Variables used by loop closing
//    long unsigned int mnLoopPointForKF;
//    long unsigned int mnCorrectedByKF;
//    long unsigned int mnCorrectedReference;
    WRITE(mp->mnBALocalForKF, file);
    WRITE(mp->mnFuseCandidateForKF, file);
    WRITE(mp->mnLoopPointForKF, file);
    WRITE(mp->mnCorrectedByKF, file);
    WRITE(mp->mnCorrectedReference, file);

//    cv::Mat mPosGBA;
    WRITEMAT(mp->mPosGBA, file);
//    long unsigned int mnBAGlobalForKF;
    WRITE(mp->mnBAGlobalForKF, file);

//    // Position in absolute coordinates
//    cv::Mat mWorldPos;
    WRITEMAT(mp->mWorldPos, file);
//
//    // Keyframes observing the point and associated index in keyframe
//    std::map<KeyFrame *, size_t> mObservations;
    size = mp->mObservations.size();
    WRITE(size, file);
    for (auto &it:mp->mObservations) {
        WRITE(it.first->mnId, file);
        WRITE(it.second, file);
    }
//    // Mean viewing direction
//    cv::Mat mNormalVector;
    WRITEMAT(mp->mNormalVector, file);
//
//    // Best descriptor to fast matching
//    cv::Mat mDescriptor;
    WRITEMAT(mp->mDescriptor, file);
//
//    // Reference KeyFrame
//    KeyFrame *mpRefKF;
    WRITE(mp->mpRefKF->mnId, file);//ID
//
//    // Tracking counters
//    int mnVisible;
//    int mnFound;
    WRITE(mp->mnVisible, file);
    WRITE(mp->mnFound, file);

//
//    // Bad flag (we do not currently erase MapPoint from memory)
//    bool mbBad;
    WRITE(mp->mbBad, file);
//    MapPoint *mpReplaced;
    WRITE(mp->mpReplaced->mnId, file);

//
//    // Scale invariance distances
//    float mfMinDistance;
//    float mfMaxDistance;
    WRITE(mp->mfMinDistance, file);
    WRITE(mp->mfMaxDistance, file);
//
//    Map *mpMap;
#ifdef SAVER_DEBUG
    char* check="mpcheck";
    fwrite(check,1,8,file);
#endif


}

void Saver::savetofile()
{
    FILE *file;
    file = fopen(filename.c_str(), "w");
    
    /*
     * Map
     */
    
    // vars in map
    WRITE(map->mnMaxKFid, file);
    WRITE(KeyFrame::nNextId, file);
    WRITE(MapPoint::nNextId, file);
    
    // map's KeyFrames
    unsigned long kfsize = map->mspKeyFrames.size();
    WRITE(kfsize, file);
    for (auto &it:map->mspKeyFrames) {
        saveOneKeyframe(it, file);
    }
    
    // map's MapPoints
    unsigned long mpsize = map->mspMapPoints.size();
    WRITE(mpsize, file);
    for (auto &it:map->mspMapPoints) {
        saveOneMapPoint(it, file);
    }
    
    // map's mvpKeyFrameOrigins
    // ONLY ID
    unsigned long kfosize = map->mvpKeyFrameOrigins.size();
    WRITE(kfosize, file);
    for (unsigned long i = 0; i < kfosize; i++) {
        long unsigned int id = map->mvpKeyFrameOrigins[i]->mnId;
        WRITE(id, file);
    }
    
    // map's mvpReferenceMapPoints
    // ONLY ID
    unsigned long rmpsize = map->mvpReferenceMapPoints.size();
    WRITE(rmpsize, file);
    for (unsigned long i = 0; i < rmpsize; i++) {
        long unsigned int id = map->mvpReferenceMapPoints[i]->mnId;
        WRITE(id, file);
    }
    
    /*
     * KFDatabase
     */
    unsigned long vsize = KB->mvInvertedFile.size();
    WRITE(vsize, file);
    for (unsigned long i = 0; i < vsize; i++) {
        unsigned long lsize = KB->mvInvertedFile[i].size();
        WRITE(lsize, file);
        for (auto &it:KB->mvInvertedFile[i]) {
            auto id = it->mnId;
            WRITE(id, file);
        }
    }
    
    fclose(file);
    printf("\nMap stored successfully!\n");
}
}