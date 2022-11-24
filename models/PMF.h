#ifndef PMF_H
#define PMF_H

#include "datamanager.h"
#include "modeltypes.h"
#include <Eigen/Dense>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_map>

namespace Model
{

using namespace std;
using namespace Eigen;

/**
 * Stores a 'snapshot' of the given theta and beta inputs by copying the inputs and storeing them in theta and beta
 * member variables.
 * @param theta A map connecting each entity ID to its corresponding latent vector.
 * @param beta A map connecting each entity ID to its corresponding latent vector.
 */
struct LatentVectorsSnapshot
{
    LatentVectorsSnapshot(const LatentVectors theta, const LatentVectors beta)
        : theta(theta)
        , beta(beta){};

    const LatentVectors theta;
    const LatentVectors beta;
};

class PMF
{
  private:
    // Initializes map vmap for each entity with random vector of size m_k sampling from distribution.
    void initVectors(gamma_distribution<> &dist, const vector<int> &entities, LatentVectors &vmap);

    // Evaluate log normal PDF at vector x.
    double logNormPDF(const VectorXd &x, double loc = 0.0, double scale = 1.0) const;

    // Evaluate log gamma PDF at vector x.
    double logGammaPDF(const VectorXd &x, double loc, double rate) const;

    // Evaluate log normal PDF at double x.
    double logNormPDF(double x, double loc = 0.0, double scale = 1.0) const;

    // Evaluate log gamma PDF at double x.
    double logGammaPDF(double x, double loc, double rate) const;

    // Evaluate log poisson PDF at double x
    double logPoisPDF(double x, double lambda) const;

    // Subset data by rows where values in column is equal to ID.
    MatrixXd subsetByID(const Ref<MatrixXd> &batch, int ID, int column) const;

    // Calculate the log probability of the data under the current model.
    void computeLoss(const LatentVectors &theta, const LatentVectors &beta);

    // Computes loss from the theta and beta snapshots found in the
    // m_loss_queue queue.
    void computeLossFromQueue();

    // Fit spot vectors to sample data in batch.
    void fitSpots(const Ref<MatrixXd> &batch, const double gamma);

    // Fit item vectors to sample data in batch.
    void fitItems(const Ref<MatrixXd> &batch, const double gamma);

    // Returns item ids of top N items recommended for given spot_id based on fitted data
    VectorXi recommend(const int spot_id, const int N) const;

    const shared_ptr<DataManager> m_data_mgr;

    // Tuneable model parameters
    const double m_lambda_theta;
    const double m_lambda_beta;
    const double m_eta_theta;
    const double m_eta_beta;
    LatentVectors m_theta;
    LatentVectors m_beta;
    const int m_k;

    // Loss computation parameters
    vector<double> m_losses;
    queue<LatentVectorsSnapshot> m_loss_queue;
    const int m_loss_interval;

    bool m_fit_in_progress;
    default_random_engine d_generator;
    mutex m_mutex;
    condition_variable m_cv;

  public:
    PMF(const shared_ptr<DataManager> &data_mgr,
        const int k,
        const double lambda_theta,
        const double lambda_beta,
        const double eta_theta,
        const double eta_beta,
        const int loss_interval);
    ~PMF();

    // Fits the exprs data sequentially updating m_theta and m_beta vectors with the learning rate given in gamma.
    // Returns the vector of loss computations computed for every 10 epochs.
    vector<double> fitSequential(const int epochs, const double gamma);

    // Fits the exprs data in parallel updating m_theta and m_beta vectors with the learning rate given in gamma.
    // This method will divide the exprs data by the given n_thread number of batches.
    // Returns the vector of loss computations computed for every 10 epochs.
    vector<double> fitParallel(const int epochs, const double gamma, const int n_threads);

    // Predicts exprs using learnt theta and beta vectors in model.
    // Input: data matrix with n rows and 2 columns (spot, item).
    // Returns a vector of predicted exprs for each spot and item.
    VectorXd predict(const MatrixXd &data) const;

    // Returns item names of top N items recommended for given spot_id based on fitted data
    vector<string> recommend(const int spot_id, const unordered_map<int, string> &item_name, const int N = 10) const;

    // Returns the top N similar items given the input item name
    vector<string> getSimilarItems(int &item_id, unordered_map<int, string> &id_name, int N = 10);

    // Get m_theta
    LatentVectors &getTheta();

    // Get m_beta
    LatentVectors &getBeta();

    // Get m_losses
    vector<double> &getComputedLoss();
};
} // namespace Model

#endif // PMF_H
