"use strict";
var DifferentialDiagnosis = (() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined")
      return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/DifferentialDiagnosis.tsx
  var DifferentialDiagnosis_exports = {};
  __export(DifferentialDiagnosis_exports, {
    default: () => DifferentialDiagnosis_default
  });
  var import_react = __toESM(__require("react"));
  var DifferentialPanel = () => {
    const [diagnoses, setDiagnoses] = (0, import_react.useState)([
      { name: "Choledocholithiasis", notes: "Common cause of obstructive jaundice" },
      { name: "Acute Cholangitis", notes: "Consider if fever develops" },
      { name: "Pancreatic Cancer", notes: "Less likely given age but needs consideration" }
    ]);
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2" }, diagnoses.map((dx, index) => /* @__PURE__ */ import_react.default.createElement("div", { key: dx.name, className: "flex items-start space-x-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col justify-center space-y-1 pt-2" }, index > 0 && /* @__PURE__ */ import_react.default.createElement("button", { className: "p-1 hover:bg-gray-100 rounded" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-gray-600" }, "\u25B2")), index < diagnoses.length - 1 && /* @__PURE__ */ import_react.default.createElement("button", { className: "p-1 hover:bg-gray-100 rounded" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-gray-600" }, "\u25BC"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex-1" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "border rounded-lg" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "p-2 bg-gray-50 rounded-t-lg" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-medium" }, "#", index + 1, ": ", dx.name)), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-2 border-t" }, dx.notes))))));
  };
  var DifferentialDiagnosis_default = DifferentialPanel;
  return __toCommonJS(DifferentialDiagnosis_exports);
})();
//# sourceMappingURL=differential.js.map
