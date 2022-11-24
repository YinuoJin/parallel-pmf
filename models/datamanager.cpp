#include <iostream>
#include <random>
#include <set>

#include "dataloader.h"
#include "datamanager.h"
#include "modeltypes.h"
#include "utils.h"

#include <gsl/gsl_assert>

namespace Model
{

namespace
{

/**
 * Centers the exprs in the given data matrix such that the midpoint of the expr scale is zero.
 * @param data The matrix to center.
 */
void zeroCenter(MatrixXd &data)
{
    using namespace Utils;

    const int expr_col = col_value(Cols::expr);
    const set<double> unique_vals{data.col(expr_col).data(),
                                  data.col(expr_col).data() + data.col(expr_col).size()};

    double sum = 0;

    for (const auto i : unique_vals)
    {
        sum += i;
    }

    const double mid = sum / unique_vals.size();

    for (int i = 0; i < data.rows(); i++)
    {
        data(i, expr_col) -= mid;
    }
}

/**
 * Randomly shuffles the rows of the given data matrix linearly spaced.
 * @param data The matrix to shuffle the rows of.
 */
void shuffle(MatrixXd &data)
{
    VectorXi ind = VectorXi::LinSpaced(data.rows(), 0, data.rows());
    shuffle(ind.data(), ind.data() + data.rows(), mt19937(random_device()()));
    data = ind.asPermutation() * data;
}

} // namespace

using namespace std;

/**
 * Initialize DataManager with shared ownership of DataLoader
 * @param data_loader shared_ptr to an instance of data_loader
 * @param ratio to split the dataset into train and test
 */
DataManager::DataManager(const shared_ptr<DataLoader> &data_loader, const double ratio)
    : m_data_loader(data_loader)
{
    loadDataset(ratio);
}

/**
 * Load the spot ids, item ids, train data, test data, splitting the dataset by the given ratio. (e.g.
 * ratio=0.7 will split to 70% train, 30% test)
 * @param ratio The ratio to split the data into training data vs. testing data.
 */
void DataManager::loadDataset(const double ratio)
{
    using namespace Utils;
    Expects(ratio > 0.0 && ratio <= 1.0);

    MatrixXd data = m_data_loader->getDataset();

    // for Poisson factorization model: avoid cetering data
    // zeroCenter(data);
    shuffle(data);

    // get all unique spot & item ids from the full dataset
    m_spots = make_shared<vector<int>>(getUnique(data, col_value(Cols::spot)));
    m_items = make_shared<vector<int>>(getUnique(data, col_value(Cols::item)));

    tie(m_data_train, m_data_test) = split(data, ratio);
}

/**
 * Splits the data rows into a train and test set by ratio (e.g. ratio=0.7 will split to 70% train, 30% test)
 * @param ratio The ratio to split the data into training data vs. testing data.
 * @return A tuple of <MatrixXd, MatrixXd> type, in which the first matrix is the training data and the second
 * matrix is the testing data.
 */
tuple<TrainingData, TestingData> DataManager::split(const MatrixXd &data, const double ratio)
{
    Expects(ratio > 0.0 && ratio <= 1.0);
    const int idx = static_cast<int>(data.rows() * ratio);

    return {make_shared<MatrixXd>(data.topRows(idx)), make_shared<MatrixXd>(data.bottomRows(data.rows() - idx))};
}

/**
 * Gets the training data set.
 * @return TestingData: a shared_ptr of the matrix of the training data set.
 */
TestingData DataManager::getTrain() const
{
    return m_data_train;
}

/**
 * Gets the testing data set.
 * @return TestingData: a shared_ptr of the matrix of the testing data set.
 */
TestingData DataManager::getTest() const
{
    return m_data_test;
}

/**
 * Gets all the unique spot ids.
 * @return a shared_ptr to the vector of the spot ids.
 */
shared_ptr<vector<int>> DataManager::getSpots() const
{
    return m_spots;
}

/**
 * Gets all the unique item ids.
 * @return a shared_ptr to the vector of the item ids.
 */
shared_ptr<vector<int>> DataManager::getItems() const
{
    return m_items;
}

} // namespace Model