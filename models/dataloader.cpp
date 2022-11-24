#include "dataloader.h"
#include "../csvlib/csv.h"
#include "utils.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

namespace Model
{

using namespace std;
using namespace Utils;

namespace
{

const tuple<string, string, string> csvheader = {"spotId", "itemId", "expr"};

/**
 * Count the total number of lines for the input file.
 * @param file_name Input file name
 * @return Number of lines of the file
 */
unsigned long int getLineNumber(const string &file_name)
{
    io::CSVReader<3> in(file_name);

    const auto [spotIdCol, itemIdCol, exprCol] = csvheader;
    in.read_header(io::ignore_extra_column, spotIdCol, itemIdCol, exprCol);
    int col1, col2;
    double col3;
    unsigned long int count = 0;

    while (in.read_row(col1, col2, col3))
    {
        count++;
    }

    return count;
}
} // namespace

DataLoader::DataLoader(const string &dataset_in, const string &res_dir)
    : m_dataset_in(dataset_in)
    , m_res_dir(res_dir)
{
}

/**
 * Loads data matrix from a csv file specified in m_dataset_in
 */
MatrixXd DataLoader::getDataset() const

{
    if (m_dataset_in.empty())
    {
        cerr << "File path to dataset not set." << endl;
        exit(1);
    }

    if (!filesystem::exists(m_dataset_in))
    {
        cerr << "Can't find the given m_dataset_in file: " << m_dataset_in << endl;
        exit(1);
    }

    // Get total line number, initialize matrix
    const unsigned long int line_count = getLineNumber(m_dataset_in);
    MatrixXd data(line_count, 3);

    // Read file content
    cout << "Loading input matrix..." << endl;

    io::CSVReader<3> in(m_dataset_in);
    const auto [spotIdCol, itemIdCol, exprCol] = csvheader;
    in.read_header(io::ignore_extra_column, spotIdCol, itemIdCol, exprCol);

    int spot_id;
    int item_id;
    double expr;
    int row_idx = 0;

    while (in.read_row(spot_id, item_id, expr))
    {
        Vector3d curr;
        curr << spot_id, item_id, expr;
        data.row(row_idx) = curr;
        row_idx++;
    }

    cout << "Finished loading input matrix." << endl;
    return data;
}

/**
 * Loads previously learnt latent theta & beta vectors from file m_res_dir
 */
tuple<LatentVectors, LatentVectors> DataLoader::getLearntVectors() const
{
    cout << "Loading previously learnt parameters into model..." << endl;

    filesystem::path theta_fname = m_res_dir / "theta.csv";
    filesystem::path beta_fname = m_res_dir / "beta.csv";

    if (!filesystem::exists(theta_fname) || !filesystem::exists(beta_fname))
    {
        cerr << "Model doesn't have learnt parameters, need to fit data first" << endl;
        exit(1);
    }

    const auto loadFromFile = [&](const filesystem::path &fpath) {
        io::CSVReader<2> in(fpath);
        in.read_header(io::ignore_extra_column, "id", "vector");
        int id;
        string str;

        LatentVectors lv;

        while (in.read_row(id, str))
        {

            vector<string> tokenized = Utils::tokenize(str);
            vector<double> vi_params(tokenized.size());
            std::transform(tokenized.begin(), tokenized.end(), vi_params.begin(),
                           [&](const string &s) { return std::stod(s); });
            Eigen::Map<VectorXd> params(vi_params.data(), vi_params.size());

            lv[id] = params;
        }
        return lv;
    };

    return {loadFromFile(theta_fname), loadFromFile(beta_fname)};
}

/**
 * Save learnt latent theta & beta vectors and computed loss to file in m_res_dir
 * @param theta Learnt theta vectors to save to file
 * @param beta Learnt beta vectors to save to file
 * @param losses Computed loss vector to save to file
 */
void DataLoader::saveTrainResults(const LatentVectors &theta, const LatentVectors &beta,
                                  const vector<double> &losses) const
{
    if (!filesystem::exists(m_res_dir))
    {
        filesystem::create_directory(m_res_dir);
    }

    cout << "Saving loss values..." << endl;
    filesystem::path loss_fname = m_res_dir / "loss.csv";
    ofstream loss_file;
    loss_file.open(loss_fname);
    loss_file << "Loss";
    for (const auto loss : losses)
    {
        loss_file << endl << loss;
    }
    loss_file.close();

    cout << "Saving model parameters..." << endl;

    const auto saveToLatentVectorsFile = [&](const filesystem::path &fpath, const LatentVectors &latentVectors) {
        ofstream file_handler;
        file_handler.open(fpath);
        file_handler << "id,vector";

        for (auto const &[id, val] : latentVectors)
        {
            file_handler << endl << id << ',' << val.transpose();
        }
        file_handler.close();
    };

    filesystem::path theta_fname = m_res_dir / "theta.csv";
    saveToLatentVectorsFile(theta_fname, theta);

    filesystem::path beta_fname = m_res_dir / "beta.csv";
    saveToLatentVectorsFile(beta_fname, beta);
}

} // namespace Model