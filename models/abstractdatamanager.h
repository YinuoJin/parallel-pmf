#ifndef FINAL_PROJECT_ABSTRACTDATAMANAGER_H
#define FINAL_PROJECT_ABSTRACTDATAMANAGER_H

#include "modeltypes.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace Model
{
using namespace std;
using namespace Eigen;

class AbstractDataManager
{
  public:
    virtual shared_ptr<MatrixXd> getTrain() const = 0;
    virtual shared_ptr<MatrixXd> getTest() const = 0;

    virtual shared_ptr<vector<int>> getSpots() const = 0;
    virtual shared_ptr<vector<int>> getItems() const = 0;

    virtual ~AbstractDataManager() = default;
};

} // namespace Model

#endif // FINAL_PROJECT_ABSTRACTDATAMANAGER_H
