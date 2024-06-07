#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;

typedef uint32_t Bitmask;

// Structure to represent an item
struct Item {
    char name = ' ';
    int weight = 0;
    int value = 0;
};

// Structure to represent a rule
struct Rule {
    string name;
    int totalWeight = 0;
    int totalValue = 0;
};

struct CombinationDetails {
    vector<int> rulesCombination;
    Bitmask chosenChars = 0;
    int remainingWeight = 0;
};

// Global variables
int containerSize;
int itemsTotalWeight = 0;
int itemsTotalValue = 0;
int maxValue = 0;
vector<Item> items;
vector<Rule> rules;
vector<Item> emptyItemsCombination;
vector<int> emptyRulesCombination;
vector<string> selectedItems;
condition_variable cv;
bool finished = false;
mutex mtxForBestResult;
vector<queue<CombinationDetails>> allCombinationDetailsQueue;
vector<mutex*> mtxForQueues;
int queueFlag = 0;
int totalNumConcurrentThreads;
int numConsumerThreads;

//// Function prototypes
bool parseProblemFile(const string inputFile, vector<Item>& items26);
bool compareRules(const Rule& rule1, const Rule& rule2);
void updateRulesWithItemWeightsAndValues(const vector<Item>& items26);
bool newValueBiggerThanMaxValue(const int newValue);
void getNumCores();
int getBitIndex(char c);
void setBit(Bitmask& chosenChars, char c);
void unsetBit(Bitmask& chosenChars, char c);
bool isBitSet(Bitmask& chosenChars, char c);
void selectItems();
void threadsInit();
void consumer(int threadID);
void generateRulesCombinationAndCalculateMaxValue(vector<int>& rulesCombination, int index, Bitmask& chosenChars, int remainingWeight);
bool isAddingNextRuleValid(const Rule& rule, Bitmask& chosenChars);
void calculateMaxValue(const vector<int>& rulesCombination, Bitmask& chosenChars, const int remainingWeight);
vector<Item> generateItemsLeft(Bitmask& chosenChars, const int remainingWeight);
void buildDPTableAndBacktrack(const int rulesCombinationValue, const int availableWeight, const vector<Item>& availableItems, const vector<int>& rulesCombination);
void updateSelectedItems(const int newValue, vector<Item>& itemsToPush, const vector<int>& rulesToPush);
void writeSelectedItemsToFile(const string outputFile);

int main() {
    // Set process priority for optimization
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    // Parse the problem file
    string inputFile = "problem.txt";
    vector<Item> items26(26);
    if (!parseProblemFile(inputFile, items26)) {
        return 1;
    }

    // Solve single threaded if no rules
    if (rules.size() == 0) {
        if (itemsTotalWeight <= containerSize) {
            if (!newValueBiggerThanMaxValue(itemsTotalValue)) {
                updateSelectedItems(itemsTotalValue, items, emptyRulesCombination);
            }
        }
        else {
            // Solve the knapsack problem with empty rules combination
            buildDPTableAndBacktrack(0, containerSize, items, emptyRulesCombination);
        }
    }
    else {
        // Sort rules based on their name
        sort(rules.begin(), rules.end(), compareRules);

        // Update rules with item weights and values
        updateRulesWithItemWeightsAndValues(items26);

        // Get hardware thread count
        getNumCores();

        if (totalNumConcurrentThreads == 1) { // Single threaded solution
            selectItems();
        }
        else { // Multi-threaded solution
            totalNumConcurrentThreads = 2; // After several experiment, 1+1 combination is the best
            numConsumerThreads = totalNumConcurrentThreads - 1;

            allCombinationDetailsQueue.resize(numConsumerThreads);
            mtxForQueues.resize(numConsumerThreads);
            for (int i = 0; i < numConsumerThreads; i++) {
                mtxForQueues[i] = new mutex();
            }

            // Set up thread pool
            threadsInit();
        }
    }

    // Write selected items to an output file
    string output_file = "output.txt";
    writeSelectedItemsToFile(output_file);

    return 0;
}

// Function to parse the problem file and populate data structures
bool parseProblemFile(const string inputFile, vector<Item>& items26) {
    ifstream file(inputFile);
    if (!file.is_open()) {
        cerr << "Error: Unable to open input file " << inputFile << endl;
        return false; // Return false if file cannot be opened
    }

    string line;
    bool parsingContainerSize = false;
    bool parsingItems = false;
    bool parsingRules = false;

    // Read each line in the file
    while (getline(file, line)) {
        istringstream iss(line);

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Identify sections in the file
        if (line == "ContainerSize:") {
            parsingContainerSize = true;
            continue;
        }

        if (line == "Items:") {
            parsingItems = true;
            parsingContainerSize = false;
            continue;
        }

        if (line == "Rules:") {
            parsingRules = true;
            parsingItems = false;
            continue;
        }

        // Parse container size
        if (parsingContainerSize) {
            iss >> containerSize;
            continue;
        }

        // Parse items
        if (parsingItems) {
            Item item;
            iss >> item.name >> item.weight >> item.value;
            items.push_back(item);

            int idx = static_cast<int>(item.name) - 65;
            items26[idx] = item; // Store items in a separate vector for faster access

            itemsTotalWeight += item.weight;
            itemsTotalValue += item.value;
        }

        // Parse rules
        if (parsingRules) {
            Rule rule;
            iss >> rule.name >> rule.totalWeight >> rule.totalValue;
            rules.push_back(rule);
        }
    }

    file.close();
    return true;
}

// Custom comparator function to sort rules by their names
bool compareRules(const Rule& rule1, const Rule& rule2) {
    return rule1.name < rule2.name;
}

// Function to update rules with item weights and values
void updateRulesWithItemWeightsAndValues(const vector<Item>& items26) {
    for (Rule& rule : rules) {
        // Find corresponding items in items26 vector
        int idx1 = static_cast<int>(rule.name[0]) - 65;
        int idx2 = static_cast<int>(rule.name[1]) - 65;

        // Add weights of corresponding items to the rule's total weight
        rule.totalWeight += items26[idx1].weight + items26[idx2].weight;

        // Add values of corresponding items to the rule's total value
        rule.totalValue += items26[idx1].value + items26[idx2].value;
    }
}

// Function to compare value with max value
bool newValueBiggerThanMaxValue(const int newValue) {
    if (newValue > maxValue) {
        return true;
    }
    return false;
}

// Function to get total number of cores
void getNumCores() {
    int num_threads = thread::hardware_concurrency();

    if (num_threads == 0) {
        totalNumConcurrentThreads = 1; // Fallback to a single thread if unable to determine hardware concurrency
    }
    else {
        totalNumConcurrentThreads = num_threads;
    }
}

// Function to get index for char
int getBitIndex(char c) {
    return c - 'A';
}

// Function to include chosen chars
void setBit(Bitmask& chosenChars, char c) {
    chosenChars |= (1 << getBitIndex(c));
}

// Function to remove chosen chars
void unsetBit(Bitmask& chosenChars, char c) {
    chosenChars &= ~(1 << getBitIndex(c));
}

// Function to to check if the char is already chosen
bool isBitSet(Bitmask& chosenChars, char c) {
    return chosenChars & (1 << getBitIndex(c));
}

// Function to generate valid combinations of rules using backtracking with pruning
void selectItems() {
    vector<int> rulesCombination; // Current combination of rules
    Bitmask chosenChars = 0;

    // Start backtracking from the first rule
    generateRulesCombinationAndCalculateMaxValue(rulesCombination, 0, chosenChars, containerSize);

    finished = true;
    cv.notify_all();

    return;
}

// Function to start all the producer and consumer threads
void threadsInit() {
    vector<thread> t;
    t.push_back(thread(selectItems));
    for (int j = 0; j < numConsumerThreads; j++) {
        t.push_back(thread(consumer, j));
    }
    for (auto& th : t) {
        th.join();
    }
}

// Function of consumer threads to calculate max value
void consumer(int threadID) {
    while (true) {
        unique_lock<mutex> lock(*mtxForQueues[threadID]);
        cv.wait(lock, [&threadID]() {
            return !allCombinationDetailsQueue[threadID].empty() || finished;
            });
        if (!allCombinationDetailsQueue[threadID].empty()) {
            CombinationDetails combinationDetail = allCombinationDetailsQueue[threadID].front();
            allCombinationDetailsQueue[threadID].pop();
            lock.unlock();
            calculateMaxValue(combinationDetail.rulesCombination, combinationDetail.chosenChars, combinationDetail.remainingWeight);
        }
        else if (finished) {
            break;
        }
    }
}

// Function to generate valid combinations of rules with items
void generateRulesCombinationAndCalculateMaxValue(vector<int>& rulesCombination, int index, Bitmask& chosenChars, int remainingWeight) {
    // Base case: All rules have been considered
    if (index == rules.size()) {
        if (totalNumConcurrentThreads == 1) {
            calculateMaxValue(rulesCombination, chosenChars, remainingWeight);
        }
        else {
            CombinationDetails combinationDetail;
            combinationDetail.rulesCombination = rulesCombination;
            combinationDetail.chosenChars = chosenChars;
            combinationDetail.remainingWeight = remainingWeight;

            unique_lock<mutex> lock(*mtxForQueues[queueFlag]);
            allCombinationDetailsQueue[queueFlag].push(combinationDetail);
            lock.unlock();
            cv.notify_one();

            if (queueFlag == (numConsumerThreads - 1)) {
                queueFlag = 0;
            }
            else {
                queueFlag++;
            }
        }

        return;
    }

    // Try not choosing the current rule
    generateRulesCombinationAndCalculateMaxValue(rulesCombination, index + 1, chosenChars, remainingWeight);

    // Try choosing the current rule if it doesn't conflict with chosen characters and doesn't exceed remaining weight
    if (isAddingNextRuleValid(rules[index], chosenChars) && rules[index].totalWeight <= remainingWeight) {
        // Add the current rule to the combination and update chosen characters
        for (char c : rules[index].name) {
            setBit(chosenChars, c);
        }
        rulesCombination.push_back(index);

        // Recur for the next rule with updated parameters
        generateRulesCombinationAndCalculateMaxValue(rulesCombination, index + 1, chosenChars, remainingWeight - rules[index].totalWeight);

        // Remove the current rule from the combination and restore chosen characters
        rulesCombination.pop_back();
        for (char c : rules[index].name) {
            unsetBit(chosenChars, c);
        }
    }
}

// Function to check if adding the current rule conflicts with chosen characters
bool isAddingNextRuleValid(const Rule& rule, Bitmask& chosenChars) {
    for (char c : rule.name) {
        if (isBitSet(chosenChars, c)) return false;
    }
    return true; // No conflict
}

// Function to calculate maximum value based on current rules combination and items left
void calculateMaxValue(const vector<int>& rulesCombination, Bitmask& chosenChars, const int remainingWeight) {
    if (rulesCombination.size() == 0) {
        // If no rule is chosen, solve the knapsack problem for items only
        buildDPTableAndBacktrack(0, containerSize, items, emptyRulesCombination);
    }
    else {
        // Calculate total value of rules combination
        int rulesCombinationTotalValue = 0;
        for (const int& idx : rulesCombination) {
            rulesCombinationTotalValue += rules[idx].totalValue;
        }

        if (remainingWeight == 0) {
            // If no more weight is available, add the value of rules combination to the total value
            if (newValueBiggerThanMaxValue(rulesCombinationTotalValue)) {
                updateSelectedItems(rulesCombinationTotalValue, emptyItemsCombination, rulesCombination);
            }
        }

        if (remainingWeight > 0) {
            // If there's still weight available, generate items left and solve knapsack problem
            vector<Item> itemsLeft = generateItemsLeft(chosenChars, remainingWeight);
            if (itemsLeft.size() == 0) {
                // If no more items can be added, add the value of rules combination to the total value
                if (newValueBiggerThanMaxValue(rulesCombinationTotalValue)) {
                    updateSelectedItems(rulesCombinationTotalValue, emptyItemsCombination, rulesCombination);
                }
            }
            else if (itemsLeft.size() == 1) {
                // If only one item is left, consider adding it along with rules combination
                if (newValueBiggerThanMaxValue(rulesCombinationTotalValue + itemsLeft[0].value)) {
                    updateSelectedItems(rulesCombinationTotalValue + itemsLeft[0].value, itemsLeft, rulesCombination);
                }
            }
            else {
                // If multiple items are left, solve knapsack problem for remaining items
                buildDPTableAndBacktrack(rulesCombinationTotalValue, remainingWeight, itemsLeft, rulesCombination);
            }
        }
    }
}

// Function to generate items left based on chosen characters and remaining weight
vector<Item> generateItemsLeft(Bitmask& chosenChars, const int remainingWeight) {
    vector<Item> itemsLeft;

    for (const Item& item : items) {
        if (isBitSet(chosenChars, item.name) == false) {
            if (item.weight <= remainingWeight) {
                itemsLeft.push_back(item);
            }
        }
    }

    return itemsLeft;
}

// Function to build the DP table and perform backtracking for knapsack problem
void buildDPTableAndBacktrack(const int rulesCombinationValue, const int availableWeight, const vector<Item>& availableItems, const vector<int>& rulesCombination) {
    int W = availableWeight;
    size_t I = availableItems.size();
    vector<vector<int>> dp(I + 1, vector<int>(W + 1)); // Dynamic programming table

    // Build dynamic programming table dp[][] in bottom-up manner
    for (int i = 1; i <= I; i++) {
        for (int w = 0; w <= W; w++) {
            if (availableItems[i - 1].weight > w) {
                dp[i][w] = dp[i - 1][w];
            }
            else {
                dp[i][w] = max(dp[i - 1][w], availableItems[i - 1].value + dp[i - 1][w - availableItems[i - 1].weight]);
            }
        }
    }

    // Backtrack to find the selected items only if more than maximum value
    if (newValueBiggerThanMaxValue(dp[I][W] + rulesCombinationValue)) {
        vector<Item> itemsCombination;

        int remainingWeight = availableWeight;
        for (size_t i = I; i > 0; i--) {
            if (dp[i][remainingWeight] != dp[i - 1][remainingWeight]) {
                itemsCombination.push_back(availableItems[i - 1]);
                remainingWeight -= availableItems[i - 1].weight;
            }
        }

        updateSelectedItems(dp[I][W] + rulesCombinationValue, itemsCombination, rulesCombination);
    }
}

// Function to push string to selectedItems vector
void updateSelectedItems(const int newValue, vector<Item>& itemsToPush, const vector<int>& rulesToPush) {
    vector<string> localSelectedItems; // Local variable to hold the selected items
    for (const int& idx : rulesToPush) {
        localSelectedItems.push_back(rules[idx].name);
    }
    for (const Item& item : itemsToPush) {
        localSelectedItems.push_back(item.name + string(""));
    }

    if (totalNumConcurrentThreads == 1) {
        maxValue = newValue;
        selectedItems = localSelectedItems;
    }
    else {
        // Update the global maxValue and selectedItems inside a critical section
        unique_lock<mutex> lock(mtxForBestResult);
        if (newValue > maxValue) {
            maxValue = newValue;
            selectedItems = localSelectedItems;
        }
        lock.unlock();
    }
}

// Function to write selected items to an output file
void writeSelectedItemsToFile(const string outputFile) {
    ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open output file " << outputFile << endl;
        return;
    }

    for (const string& item : selectedItems) {
        outFile << item << endl;
    }
    outFile.close();
}
