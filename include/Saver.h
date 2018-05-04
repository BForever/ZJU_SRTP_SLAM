//
// Created by 范宏昌 on 2018/4/13.
//
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "pointcloudmapping.h"


#ifndef ORB_SLAM2_SAVER_H
#define ORB_SLAM2_SAVER_H
namespace ORB_SLAM2 {
class KeyFrameDatabase;


class Saver {
public:
    Saver(string SLAMsrcfile);
    
    void setclasses(KeyFrameDatabase *pKeyFrameDatabase, Map *pmap, ORBVocabulary *vocabulary);
    
    void loadfromfile();
    
    void savetofile();

private:
    string filename;
    
    Map *map;
    KeyFrameDatabase *KB;
    std::map<long unsigned int, KeyFrame *> kfmap;
    std::map<long unsigned int, MapPoint *> mpmap;
    std::vector<KeyFrame*> kfv;
    std::vector<MapPoint*> mpv;
    ORBVocabulary *mpORBvocabulary;
    
    void saveOneKeyframe(KeyFrame *kf, FILE *file);
    
    void saveOneMapPoint(MapPoint *mp, FILE *file);
    
    void loadOneKeyframe(FILE *file);
    
    void loadOneMapPoint(FILE *file);
    
    void replaceAllIndex();
    
    //KF
    std::vector<std::vector<long unsigned int> > mvpMapPoints_id_all;
    std::vector<std::vector<long unsigned int>> mConnectedKeyFrameWeights_id_all;
    std::vector<std::vector<int>> mConnectedKeyFrameWeights_weight_all;
    std::vector<std::vector<long unsigned int>> mvpOrderedConnectedKeyFrames_id_all;
    std::vector<std::vector<long unsigned int>> mspChildrens_id_all;
    std::vector<std::vector<long unsigned int>> mspLoopEdges_id_all;
    std::vector<long unsigned int> mpParent_id_all;
    
    //MP
    std::vector<std::vector<long unsigned int> > mObservations_id_all;
    std::vector<std::vector<size_t> > mObservations_size_t_all;
    std::vector<long unsigned int>  mpRefKF_id_all;
    std::vector<long unsigned int> mpReplaced_id_all;
};


}
#endif //ORB_SLAM2_SAVER_H
