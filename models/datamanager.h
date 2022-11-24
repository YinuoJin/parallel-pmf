#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include "abstractdatamanager.h"
#include "modeltypes.h"
#include <Eigen/Dense>
#include <memory>
#include <tuple>

namespace Model
{
class DataLoader;
} // namespace Model

namespace Model
{

using namespace std;
using namespace Eigen;

class DataManager : public AbstractDataManager
{
  private:
    const shared_ptr<DataLoader> m_data_loader;

    shared_ptr<MatrixXd> m_data_train;
    shared_ptr<MatrixXd> m_data_test;

    shared_ptr<vector<int>> m_spots;
    shared_ptr<vector<int>> m_items;

    // Splits data into train-test data with given ratio (e.g. ratio=0.7 will split to 70% train, 30% test)
    tuple<TrainingData, TestingData> split(const MatrixXd &data, const double ratio);

  public:
    DataManager(const shared_ptr<DataLoader> &data_loader, const double ratio);

    virtual void loadDataset(const double ratio);

    virtual TrainingData getTrain() const;
    virtual TestingData getTest() const;

    virtual shared_ptr<vector<int>> getSpots() const;
    virtual shared_ptr<vector<int>> getItems() const;
};

} // namespace Model

#endif // FINAL_PROJECT_DATAMANAGER_H
