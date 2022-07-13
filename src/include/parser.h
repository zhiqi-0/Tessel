#pragma once


#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>


enum class OptionAction { StoreTrue, KeyValue};


class Option {

 public:
    
    Option(std::string name, OptionAction action):
      _name(name), _set_default(false), _set_val(false), _act(action) {}

    void setDefault(const std::string& val) {
        _default = val;
        _set_default = true;
    }

    bool hasDefault() { return _set_default; }

    void setVal(const std::string& val) {
        _val = val;
        _set_val = true;
    }

    bool hasVal() { return _set_val; }

    std::string getStr() {
        if (hasVal()) {
            return _val;
        }
        else if (hasDefault()) {
            return _default;
        }
        else {
            std::runtime_error("default and value not set.\n");
        }
    }

    template <typename T>
    T get() {
        T ret;
        std::stringstream ss;
        if (hasVal()) {
            if (!(ss << _val and ss >> ret and ss.eof())) {
                throw std::bad_cast();
            }
            return ret;
        }
        else if (hasDefault()) {
            if (!(ss << _default and ss >> ret and ss.eof())) {
                throw std::bad_cast();
            }
            return ret;
        }
        else {
            std::runtime_error("default and value not set.\n");
        }
    }


 private:

    std::string _name;
    bool _set_default;
    std::string _default;
    bool _set_val;
    std::string _val;

    OptionAction _act;

};


class CmdParser {

 public:
    
    CmdParser() {}
    ~CmdParser() {
        for (auto& it : _field) {
            delete it.second;
        }
    }

    template <typename T>
    void add(const std::string& name, const std::string& help="",
             OptionAction action = OptionAction::KeyValue) {
        if (_field.find(name) != _field.end()) {
            throw std::runtime_error(name + " double defined\n");
        }
        _field.emplace(name, new Option(name, action));
    }

    template <typename T>
    void setDefault(const std::string& name, T val) {
        if (!this->exist(name)) {
            this->add<T>(name);
        }
        _field[this->name2field(name)]->setDefault(std::to_string(val));
    }

    inline std::string name2field(std::string name) {
        if (name.substr(0, 2) == "--") {
            return name;
        }
        else {
            return std::string("--") + name;
        }
    }

    inline bool exist(const std::string& name) {
        std::string field = name2field(name);
        return _field.find(field) != _field.end();
    }

    std::vector<std::string> names() const {
        std::vector<std::string> ret;
        for (auto it : _field) {
            ret.push_back(it.first.substr(2, it.first.size()-2));
        }
        return ret;
    }

    void parse(int argc, const char* argv[]) {
        if (argc < 1) return;
        int idx = 1;
        while (idx < argc) {
            std::string name(argv[idx]);
            if (name.size() > 2 and name.substr(0, 2) == "--") {
                if (!this->exist(name)) {
                    throw std::runtime_error(name + " : not found in argument list\n");
                }
                if (idx + 1 < argc) {
                    _field[name]->setVal(std::string(argv[idx+1]));
                    idx += 2;
                }
            }
            else {
                throw std::runtime_error("expected field starting with --.\n");
            }
        }
    }

    template <typename T>
    T get(const std::string& name) {
        std::string field = name2field(name);
        if (!this->exist(field)) {
            throw std::runtime_error(field + " : field not found\n");
        }
        return _field[field]->get<T>();
    }

    template <typename T>
    T get(const char* name) {
        return this->get<T>(std::string(name));
    }

    std::string getStr(const std::string& name) {
        if (!this->exist(name)) {
            throw std::runtime_error(name + " : field not found\n");
        }
        std::string field = name2field(name);
        return _field[field]->getStr();
    }

    friend std::ostream& operator<<(std::ostream& out, CmdParser& parser) {
        for (auto& name : parser.names()) {
            out << "parser field: " << name << " val: " << parser.getStr(name) << std::endl;
        }
        return out;
    }

 private:

    std::unordered_map<std::string, Option*> _field;

};