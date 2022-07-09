#pragma once


class UniqueID {

 private:

    static UniqueID* uinstance;
    int uid = 0;
    UniqueID() {}
    UniqueID(const UniqueID & obj) {}


 public:

    static UniqueID* GetInstance() {
        if (uinstance == nullptr) {
            uinstance = new UniqueID();
        }
        return uinstance;
    }

    int GetUid() {
        uid += 1;
        return uid;
    }

    ~UniqueID() {}
};

UniqueID* UniqueID::uinstance = nullptr;