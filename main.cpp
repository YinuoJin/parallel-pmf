#include "models/PMF.h"
#include "models/dataloader.h"
#include "models/datamanager.h"
#include "models/utils.h"
#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace std;
using namespace Model;
using namespace Utils;
using namespace chrono;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // clang-format off

    po::options_description d   esc("Parameters for Probabilistic Matrix Factorization (PMF)");

    desc.add_options()
            ("help,h", "Help")
            ("input,i", po::value<string>(), "Input file name")
            ("map,m", po::value<string>(), "Item mapping file name")
            ("use_defaults,d", po::bool_switch()->default_value(false), "If enabled, uses './movielens/ratings.csv' for the input file and './movielens/movies.csv' for the map input file")
            ("output,o", po::value<filesystem::path>(), "Output directory\n[default: current_path/results/]\n\n")
            ("task", po::value<string>()->default_value("train"), "Task to perform\n[Options: 'train', 'recommend']\n")
            ("n_components,k", po::value<int>()->default_value(5), "Number of components (k)\n[default: 3]\n")
            ("n_epochs,n", po::value<int>()->default_value(200), "Num. of learning iterations\n[default: 200]\n")
            ("ratio,r", po::value<double>()->default_value(0.7), "Ratio for training/test set splitting\n[default: 0.7]\n")
            ("thread", po::value<int>()->default_value(4), "Number of threads for parallelization\nThis value must be at least 2\n[default: 4]\n")
            ("gamma", po::value<double>()->default_value(0.01), "Learning rate for gradient descent\n[default: 0.01]\n")
            ("std_theta", po::value<double>()->default_value(1), "Std. of theta's prior normal distribution\n[default: 1]\n")
            ("std_beta", po::value<double>()->default_value(1), "Std. of beta's prior normal distribution\n[default: 1]\n")
            ("run_sequential,s", po::bool_switch()->default_value(false), "Enable running model fitting sequentially\n")
            ("user", po::bool_switch()->default_value(false), "Recommend items for given user\n")
            ("item", po::bool_switch()->default_value(false), "Recommend similar items for a given item\n")
            ("loss_interval, l", po::value<int>()->default_value(10), "Number of epochs between each loss computation.\n[default: 10]");

    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        cout << desc << endl;
        return 0;
    }

    // Parse program options
    Arguments args = Utils::ArgParser(argc, argv, vm, desc);

    // (1). read CSV & split to training & test sets
    auto dm_t0 = chrono::steady_clock::now();

    const auto data_loader = make_shared<DataLoader>(args.indir, args.outdir);
    const auto data_mgr = make_shared<DataManager>(data_loader, args.ratio);

    auto dm_t1 = chrono::steady_clock::now();
    double dm_delta_t = std::chrono::duration<double, std::milli>(dm_t1 - dm_t0).count() * 0.001;
    cout << "Took " << dm_delta_t << " s. to load data. \n\n";

    // (2). Fit the model to the training data
    Model::PMF model{data_mgr, args.k, args.std_beta, args.std_theta, args.loss_interval};

    if (args.task == "train")
    {
        // (2.1) Model training
        auto fit_t0 = chrono::steady_clock::now();

        vector<double> losses;

        if (args.run_fit_sequential)
        {
            losses = model.fitSequential(args.n_epochs, args.gamma);
        }
        else
        {
            losses = model.fitParallel(args.n_epochs, args.gamma, args.n_threads);
        }

        auto fit_t1 = chrono::steady_clock::now();

        double fit_delta_t = std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count() * 0.001;
        cout << "Running time for " << args.n_epochs << " iterations: " << fit_delta_t << " s.\n\n";

        // (2.2) Evaluate model quality on test data
        shared_ptr<MatrixXd> ratings_test = data_mgr->getTest();
        VectorXd actual = ratings_test->rightCols(1);
        VectorXd predicted = model.predict(ratings_test->leftCols(2));

        double error = Utils::rmse(actual, predicted);
        double baseline_zero = Utils::rmse(actual, 0.0);
        double baseline_avg = Utils::rmse(actual, actual.mean());

        cout << "RMSE(0): " << baseline_zero << endl;
        cout << "RMSE(mean): " << baseline_avg << endl;
        cout << "RMSE(pred): " << error << endl;

        // (3.2) save loss & trained parameters to file
        data_loader->saveTrainResults(model.getTheta(), model.getBeta(), model.getComputedLoss());
    }
    else if (args.task == "recommend")
    {
        // (3). Recommendations
        // (3.1) Load model from file
        tie(model.getTheta(), model.getBeta()) = data_loader->getLearntVectors();
        Utils::ItemMap item_map = Utils::loadItemMap(args.mapdir);

        if (args.rec_option == RecOption::user)
        {
            // (3.2-1) recommend items for user
            cout << "Please specify user id: " << endl;
            int user_id;
            cin >> user_id;

            if (item_map.id_name.find(user_id) == item_map.id_name.end())
            {
                cerr << "User id " << user_id << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.recommend(user_id, item_map.id_name, 10);
                cout << "\nTop 10 recommended items for user " << user_id << " :" << endl << endl;
                for (auto &item : rec)
                {
                    cout << "Item: " << item << '\t' << "Attribute: " << item_map.name_item_attributes[item] << endl;
                }
            }
        }
        else
        {
            // (3.2-2) recommend similar items from item
            cout << "Please specify item name: " << endl;
            string item_name;
            getline(cin, item_name);

            if (item_map.name_id.find(item_name) == item_map.name_id.end())
            {
                cerr << "Item " << item_name << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.getSimilarItems(item_map.name_id[item_name], item_map.id_name, 10);
                cout << "\nTop 10 similar items to " << item_name << " :" << endl << endl;
                for (auto &item : rec)
                {
                    cout << "Item: " << item << '\t' << "Attributes: " << item_map.name_item_attributes[item] << endl;
                }
            }
        }
    }

    return 0;
}
