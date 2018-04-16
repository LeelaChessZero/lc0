#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "utils/exception.h"

namespace lczero {

class UciOptions {
 public:
  UciOptions(int argc, const char** argv) : argc_(argc), argv_(argv) {}

  class Option {
   public:
    Option(const std::string& name, const std::string& long_flag,
           char short_flag);
    virtual ~Option(){};
    // Set value from string.
    virtual void SetValue(const std::string& value) = 0;
    // If has integer value, return, otherwise throw exception.
    virtual int GetIntValue() const;
    // If has string value, return, otherwise throw exception.
    virtual std::string GetStringValue() const;
    // If has boolean value, return, otherwise throw exception.
    virtual bool GetBoolValue() const;

   protected:
    virtual const std::string& GetName() const { return name_; }
    const std::string& GetLongFlag() const { return long_flag_; }
    char GetShortFlag() const { return short_flag_; }

   private:
    virtual std::string GetOptionString() const = 0;
    virtual void SendValue() const = 0;
    virtual bool ProcessLongFlag(const std::string& flag,
                                 const std::string& value) {
      return false;
    }
    virtual bool ProcessShortFlag(char flag) { return false; }
    virtual bool ProcessShortFlagWithValue(char flag,
                                           const std::string& value) {
      return false;
    }
    virtual std::string GetHelp() const = 0;

    std::string name_;
    std::string long_flag_;
    char short_flag_;
    friend class UciOptions;
  };

  // Add an option to the list of available options (from command line flags
  // or UCI params)
  void Add(std::unique_ptr<Option> option);

  // Returns list of options in UCI format.
  std::vector<std::string> ListOptionsUci() const;

  // Set the option from string value.
  void SetOption(const std::string& name, const std::string& value);
  // Call option setter for this option.
  void SendOption(const std::string& name);
  // Call option setter all options.
  void SendAllOptions();
  // Gets option by name.
  const Option* GetOption(const std::string& name) const;
  // Get option's int value or throws if value is not int.
  int GetIntValue(const std::string& name) const;
  // Get option's bool value or throws if value is not int.
  bool GetBoolValue(const std::string& name) const;
  // Processes all flags. Returns false if should exit.
  bool ProcessAllFlags();

  std::string GetProgramName() const { return argv_[0]; }

 private:
  void ShowHelp() const;

  Option* FindOptionByName(const std::string& name) const;
  std::vector<std::unique_ptr<Option>> options_;
  int argc_;
  const char** argv_;
};

class StringOption : public UciOptions::Option {
 public:
  StringOption(const std::string& name, const std::string& def,
               std::function<void(const std::string&)> setter,
               const std::string& long_flag = {}, char short_flag = '\0');
  void SetValue(const std::string& value) override { value_ = value; }
  std::string GetStringValue() const override { return value_; }

 private:
  std::string GetOptionString() const override;
  void SendValue() const override;
  bool ProcessLongFlag(const std::string& flag,
                       const std::string& value) override;
  std::string GetHelp() const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value) override;

  std::string value_;
  std::function<void(const std::string&)> setter_;
};

class SpinOption : public UciOptions::Option {
 public:
  SpinOption(const std::string& name, int def, int min, int max,
             std::function<void(int)> setter, const std::string& long_flag = {},
             char short_flag = '\0');
  void SetValue(const std::string& value) override;
  int GetIntValue() const override;

 private:
  std::string GetOptionString() const override;
  void SendValue() const override;
  bool ProcessLongFlag(const std::string& flag,
                       const std::string& value) override;
  std::string GetHelp() const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value) override;

  int value_;
  int min_;
  int max_;
  std::function<void(int)> setter_;
};

class CheckOption : public UciOptions::Option {
 public:
  CheckOption(const std::string& name, bool def,
              std::function<void(bool)> setter,
              const std::string& long_flag = {}, char short_flag = '\0');
  void SetValue(const std::string& value) override;
  bool GetBoolValue() const override { return value_; }

 private:
  std::string GetOptionString() const override;
  void SendValue() const override;
  bool ProcessLongFlag(const std::string& flag,
                       const std::string& value) override;
  std::string GetHelp() const override;
  bool ProcessShortFlag(char flag) override;

  bool value_;
  std::function<void(bool)> setter_;
};

}  // namespace lczero