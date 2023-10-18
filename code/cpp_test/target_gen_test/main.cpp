#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>

typedef std::vector<std::string> StringList;
typedef std::map<std::string, int> StringToIntMap;

std::pair<StringList, StringToIntMap> findClasses(const std::string& directory) {
    StringList classes;
    StringToIntMap classToIdx;

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_directory() && entry.path().filename() != "ILSVRC2012_img_val") {
            classes.push_back(entry.path().filename());
        }
    }

    std::sort(classes.begin(), classes.end());

    if (classes.empty()) {
        throw std::runtime_error("Couldn't find any class folder in " + directory);
    }

    int index = 0;
    for (const std::string& cls : classes) {
        classToIdx[cls] = index;
        index++;
    }

    return std::make_pair(classes, classToIdx);
}

int main() {
    std::string directory = "/data/imagenet/val";
    try {
        auto result = findClasses(directory);
        StringList classes = result.first;
        StringToIntMap classToIdx = result.second;

        for (const auto& cls : classes) {
            std::cout << cls << " : " << classToIdx[cls] << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
