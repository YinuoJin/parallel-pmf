//
// Created by yinuo on 3/14/21.
//

#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "csvlib/csv.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

/**
 * Load and centralize rating matrix
 *
 * @param input path to input data file
 * @return processed matrix
 */
MatrixXd loadData(const string &input)
{
    // todo: the number of cols to take in the current CSV parser isn't flexible; no need to worry if we always read 3 cols
    if (!fs::exists(input))
    {
        cerr << "Can't find the given input file: " << input << endl;
        exit(1);
    }
    cout << "Loading input matrix..." << endl;
    io::CSVReader<3> in(input);
    in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
    int user_id;
    int movie_id;
    double rating;
    MatrixXd ratings(1, 3);

    while (in.read_row(user_id, movie_id, rating))
    {
        Vector3d curr;
        curr << user_id, movie_id, rating;
        ratings.row(ratings.rows() - 1) = curr;
        ratings.conservativeResize(ratings.rows() + 1, ratings.cols());
    }
    ratings.conservativeResize(ratings.rows() - 1, ratings.cols());

    // center ratings to mean = 0
    set<double> unique_rates{ratings.col(2).data(), ratings.col(2).data() + ratings.col(2).size()};
    double sum = 0;
    for (auto i : unique_rates)
    {
        sum += i;
    }
    double mid = sum / unique_rates.size();
    for (int i = 0; i < ratings.rows(); i++)
    {
        ratings(i, 2) -= mid;
    }

    return ratings;
}

int main(int argc, char **argv)
{
    // parse arguments, path configuration
    string input;
    fs::path outdir("results");
    int k;
    int n_epochs = 200;  // default # of iterations
    double gamma = 0.01; // default learning rate for gradient descent

    po::options_description desc("Parameters for Probabilistic Matrix Factorization (PMF)");
    desc.add_options()("help,h", "Help")("input,i", po::value<string>(&input), "Input file name")("output,o", po::value<fs::path>(&outdir), "Output directory\n  [default: current_path/results/]")("n_components,k", po::value<int>(&k), "Number of components (k)")("n_epochs,n", po::value<int>(&n_epochs), "Num. of learning iterations\n  [default: 200]")("gamma", po::value<double>(&gamma), "learning rate for gradient descent\n  [default: 1e-2]");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        cout << desc << endl;
        return 0;
    }
    if (outdir.empty())
    {
        outdir = "results";
    }
    if (!fs::exists(outdir))
    {
        cout << "Outdir " << outdir << " exists" << endl;
    }
    else
    {
        cout << "Outdir doesn't exist, creating " << outdir << "..." << endl;
        fs::create_directory(outdir);
    }

    // (1). read CSV & save to matrix object
    MatrixXd ratings = loadData(input);

    // (2). todo: split matrix into training & validation sets

    // (3). todo: implement PMF class

    // (4). todo: training

    // (5). todo: output losses & prediction results to outdir, write python scripts for visualization & other calculations
    return 0;
}
