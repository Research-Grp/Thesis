// Copyright 2012 Google Inc. All rights reserved.
(function() {

    var data = {
        "resource": {
            "version": "1",

            "macros": [{ "function": "__e" }, { "function": "__aev", "vtp_varType": "URL", "vtp_component": "IS_OUTBOUND", "vtp_affiliatedDomains": ["list"] }, { "function": "__v", "vtp_name": "gtm.triggers", "vtp_dataLayerVersion": 2, "vtp_setDefaultValue": true, "vtp_defaultValue": "" }, { "function": "__v", "vtp_name": "gtm.elementId", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.elementClasses", "vtp_dataLayerVersion": 1 }, { "function": "__aev", "vtp_varType": "URL", "vtp_component": "URL_NO_FRAGMENT" }, { "function": "__aev", "vtp_varType": "URL", "vtp_component": "HOST", "vtp_stripWww": true }, { "function": "__aev", "vtp_varType": "URL", "vtp_component": "EXTENSION" }, { "function": "__aev", "vtp_varType": "TEXT" }, { "function": "__aev", "vtp_varType": "URL", "vtp_component": "PATH" }, { "function": "__v", "vtp_name": "gtm.videoStatus", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoUrl", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoTitle", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoProvider", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoCurrentTime", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoDuration", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoPercent", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.videoVisible", "vtp_dataLayerVersion": 1 }, { "function": "__u", "vtp_component": "QUERY", "vtp_queryKey": "q,s,search,query,keyword", "vtp_multiQueryKeys": true, "vtp_ignoreEmptyQueryParam": true, "vtp_enableMultiQueryKeys": false, "vtp_enableIgnoreEmptyQueryParam": false }, { "function": "__v", "vtp_name": "gtm.scrollThreshold", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.historyChangeSource", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.oldUrl", "vtp_dataLayerVersion": 1 }, { "function": "__v", "vtp_name": "gtm.newUrl", "vtp_dataLayerVersion": 1 }, { "function": "__c", "vtp_value": "google.com.ph" }, { "function": "__c", "vtp_value": 0 }, { "vtp_signal": 0, "function": "__c", "vtp_value": 0 }],
            "tags": [{ "function": "__gct", "vtp_trackingId": "G-CD35DZJ728", "vtp_sessionDuration": 0, "tag_id": 1 }, { "function": "__get", "vtp_eventName": "click", "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "link_id", ["macro", 3], "link_classes", ["macro", 4], "link_url", ["macro", 5], "link_domain", ["macro", 6], "outbound", true]], "vtp_deferrable": false, "tag_id": 11 }, { "function": "__get", "vtp_eventName": "file_download", "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "link_id", ["macro", 3], "link_text", ["macro", 8], "link_url", ["macro", 5], "file_name", ["macro", 9], "file_extension", ["macro", 7]]], "vtp_deferrable": false, "tag_id": 18 }, { "function": "__get", "vtp_eventName": ["template", "video_", ["macro", 10]], "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "video_url", ["macro", 11], "video_title", ["macro", 12], "video_provider", ["macro", 13], "video_current_time", ["macro", 14], "video_duration", ["macro", 15], "video_percent", ["macro", 16], "visible", ["macro", 17]]], "vtp_deferrable": false, "tag_id": 21 }, { "function": "__get", "vtp_eventName": "view_search_results", "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "search_term", ["macro", 18]]], "vtp_deferrable": true, "tag_id": 26 }, { "function": "__get", "vtp_eventName": "scroll", "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "percent_scrolled", ["macro", 19]]], "vtp_deferrable": false, "tag_id": 29 }, { "function": "__get", "vtp_eventName": "page_view", "vtp_settings": ["map", "streamId", "G-CD35DZJ728", "eventParameters", ["map", "page_referrer", ["macro", 21]]], "vtp_deferrable": false, "tag_id": 32 }, { "function": "__dlm", "vtp_userInput": ["list", ["map", "key", "gtm.gtagReferrer.G-CD35DZJ728", "value", ["macro", 21]]], "tag_id": 33 }, { "function": "__set_product_settings", "original_activity_id": 3001, "vtp_foreignTldMacroResult": ["macro", 23], "vtp_isChinaVipRegionMacroResult": ["macro", 24], "tag_id": 36 }, { "function": "__ogt_event_settings", "original_activity_id": 3002, "vtp_eventSettings": ["list", ["map", "name", "purchase", "conversion", true]], "tag_id": 37 }, { "function": "__ogt_google_signals", "original_activity_id": 3003, "vtp_googleSignals": "DISABLED", "vtp_serverMacroResult": ["macro", 25], "tag_id": 38 }, { "function": "__lcl", "vtp_waitForTags": false, "vtp_checkValidation": true, "vtp_uniqueTriggerId": "1_9", "tag_id": 39 }, { "function": "__lcl", "vtp_waitForTags": false, "vtp_checkValidation": true, "vtp_uniqueTriggerId": "1_16", "tag_id": 40 }, { "function": "__ytl", "vtp_captureStart": true, "vtp_captureComplete": true, "vtp_captureProgress": true, "vtp_progressThresholdsPercent": "10,25,50,75", "vtp_triggerStartOption": "DOM_READY", "vtp_uniqueTriggerId": "1_19", "vtp_enableTriggerStartOption": true, "tag_id": 41 }, { "function": "__sdl", "vtp_verticalThresholdUnits": "PERCENT", "vtp_verticalThresholdsPercent": "90", "vtp_verticalThresholdOn": true, "vtp_horizontalThresholdOn": false, "vtp_triggerStartOption": "WINDOW_LOAD", "vtp_uniqueTriggerId": "1_27", "vtp_enableTriggerStartOption": true, "tag_id": 42 }, { "function": "__ehl", "vtp_groupEvents": true, "vtp_groupEventsInterval": 1000, "vtp_uniqueTriggerId": "1_30", "tag_id": 43 }],
            "predicates": [{ "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.js" }, { "function": "_eq", "arg0": ["macro", 1], "arg1": true }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.linkClick" }, { "function": "_re", "arg0": ["macro", 2], "arg1": "(^$|((^|,)1_9($|,)))" }, { "function": "_re", "arg0": ["macro", 7], "arg1": "pdf|xlsx?|docx?|txt|rtf|csv|exe|key|pp(s|t|tx)|7z|pkg|rar|gz|zip|avi|mov|mp4|mpe?g|wmv|midi?|mp3|wav|wma", "ignore_case": true }, { "function": "_re", "arg0": ["macro", 2], "arg1": "(^$|((^|,)1_16($|,)))" }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.video" }, { "function": "_re", "arg0": ["macro", 2], "arg1": "(^$|((^|,)1_19($|,)))" }, { "function": "_eq", "arg0": ["macro", 18], "arg1": "undefined" }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.scrollDepth" }, { "function": "_re", "arg0": ["macro", 2], "arg1": "(^$|((^|,)1_27($|,)))" }, { "function": "_eq", "arg0": ["macro", 20], "arg1": ["list", "pushState", "popstate", "replaceState"], "any_of": true }, { "function": "_eq", "arg0": ["macro", 21], "arg1": ["macro", 22] }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.historyChange-v2" }, { "function": "_re", "arg0": ["macro", 2], "arg1": "(^$|((^|,)1_30($|,)))" }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.init" }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.dom" }, { "function": "_eq", "arg0": ["macro", 0], "arg1": "gtm.load" }],
            "rules": [
                [
                    ["if", 0],
                    ["add", 0, 11, 12, 15]
                ],
                [
                    ["if", 1, 2, 3],
                    ["add", 1]
                ],
                [
                    ["if", 2, 4, 5],
                    ["add", 2]
                ],
                [
                    ["if", 6, 7],
                    ["add", 3]
                ],
                [
                    ["if", 0],
                    ["unless", 8],
                    ["add", 4]
                ],
                [
                    ["if", 9, 10],
                    ["add", 5]
                ],
                [
                    ["if", 11, 13, 14],
                    ["unless", 12],
                    ["add", 6, 7]
                ],
                [
                    ["if", 15],
                    ["add", 8, 9, 10]
                ],
                [
                    ["if", 16],
                    ["add", 13]
                ],
                [
                    ["if", 17],
                    ["add", 14]
                ]
            ]
        },
        "runtime": [
            [50, "__set_product_settings", [46, "a"],
                [52, "b", ["require", "internal.setProductSettingsParameter"]],
                [52, "c", ["require", "getContainerVersion"]],
                [52, "d", [17, ["c"], "containerId"]],
                ["b", [15, "d"], "google_tld", [17, [15, "a"], "foreignTldMacroResult"]],
                ["b", [15, "d"], "ga_restrict_domain", [20, [17, [15, "a"], "isChinaVipRegionMacroResult"], 1]],
                [2, [15, "a"], "gtmOnSuccess", [7]]
            ],
            [50, "__ogt_event_settings", [46, "a"],
                [52, "b", ["require", "internal.setProductSettingsParameter"]],
                [52, "c", ["require", "getContainerVersion"]],
                [52, "d", [8]],
                [53, [41, "f"],
                    [3, "f", 0],
                    [63, [7, "f"],
                        [23, [15, "f"],
                            [17, [17, [15, "a"], "eventSettings"], "length"]
                        ],
                        [33, [15, "f"],
                            [3, "f", [0, [15, "f"], 1]]
                        ],
                        [46, [53, [52, "g", [16, [16, [17, [15, "a"], "eventSettings"],
                                [15, "f"]
                            ], "name"]],
                            [43, [15, "d"],
                                [15, "g"],
                                [8, "blacklisted", [16, [16, [17, [15, "a"], "eventSettings"],
                                    [15, "f"]
                                ], "blacklisted"], "conversion", [16, [16, [17, [15, "a"], "eventSettings"],
                                    [15, "f"]
                                ], "conversion"]]
                            ]
                        ]]
                    ]
                ],
                [52, "e", [17, ["c"], "containerId"]],
                ["b", [15, "e"], "event_settings", [15, "d"]],
                [2, [15, "a"], "gtmOnSuccess", [7]]
            ],
            [50, "__ogt_google_signals", [46, "a"],
                [52, "b", ["require", "internal.setProductSettingsParameter"]],
                [52, "c", ["require", "getContainerVersion"]],
                [52, "d", [13, [41, "$0"],
                    [3, "$0", ["require", "internal.getFlags"]],
                    ["$0"]
                ]],
                [52, "e", [17, ["c"], "containerId"]],
                ["b", [15, "e"], "google_signals", [20, [17, [15, "a"], "serverMacroResult"], 1]],
                [22, [17, [15, "d"], "enableGa4OnoRemarketing"],
                    [46, ["b", [15, "e"], "google_ono", [20, [17, [15, "a"], "serverMacroResult"], 2]]]
                ],
                [2, [15, "a"], "gtmOnSuccess", [7]]
            ]
        ],
        "permissions": { "__set_product_settings": { "read_container_data": {} }, "__ogt_event_settings": { "read_container_data": {} }, "__ogt_google_signals": { "read_container_data": {} } }

        ,
        "security_groups": {
            "google": ["__set_product_settings", "__ogt_event_settings", "__ogt_google_signals"]
        }

    };


    /*
    
     Copyright The Closure Library Authors.
     SPDX-License-Identifier: Apache-2.0
    */
    var l, aa = function(a) { var b = 0; return function() { return b < a.length ? { done: !1, value: a[b++] } : { done: !0 } } },
        ca = function(a) { return a.raw = a },
        da = function(a) { var b = "undefined" != typeof Symbol && Symbol.iterator && a[Symbol.iterator]; return b ? b.call(a) : { next: aa(a) } },
        ea = "function" == typeof Object.create ? Object.create : function(a) { var b = function() {};
            b.prototype = a; return new b },
        fa;
    if ("function" == typeof Object.setPrototypeOf) fa = Object.setPrototypeOf;
    else { var ha;
        a: { var ia = { a: !0 },
                ja = {}; try { ja.__proto__ = ia;
                ha = ja.a; break a } catch (a) {}
            ha = !1 }
        fa = ha ? function(a, b) { a.__proto__ = b; if (a.__proto__ !== b) throw new TypeError(a + " is not extensible"); return a } : null }
    var ka = fa,
        la = function(a, b) { a.prototype = ea(b.prototype);
            a.prototype.constructor = a; if (ka) ka(a, b);
            else
                for (var c in b)
                    if ("prototype" != c)
                        if (Object.defineProperties) { var d = Object.getOwnPropertyDescriptor(b, c);
                            d && Object.defineProperty(a, c, d) } else a[c] = b[c];
            a.Jk = b.prototype },
        ma = this || self,
        na = function(a) { return a };
    var oa = function(a, b) { this.g = a;
        this.o = b };
    var pa = function(a) { return "number" === typeof a && 0 <= a && isFinite(a) && 0 === a % 1 || "string" === typeof a && "-" !== a[0] && a === "" + parseInt(a, 10) },
        qa = function() { this.C = {};
            this.s = !1;
            this.I = {} },
        ra = function(a, b) { var c = [],
                d; for (d in a.C)
                if (a.C.hasOwnProperty(d)) switch (d = d.substr(5), b) {
                    case 1:
                        c.push(d); break;
                    case 2:
                        c.push(a.get(d)); break;
                    case 3:
                        c.push([d, a.get(d)]) }
                return c };
    qa.prototype.get = function(a) { return this.C["dust." + a] };
    qa.prototype.set = function(a, b) { this.s || (a = "dust." + a, this.I.hasOwnProperty(a) || (this.C[a] = b)) };
    qa.prototype.has = function(a) { return this.C.hasOwnProperty("dust." + a) };
    var sa = function(a, b) { b = "dust." + b;
        a.s || a.I.hasOwnProperty(b) || delete a.C[b] };
    qa.prototype.Gb = function() { this.s = !0 };
    var ta = function(a) { this.o = new qa;
        this.g = [];
        this.s = !1;
        a = a || []; for (var b in a) a.hasOwnProperty(b) && (pa(b) ? this.g[Number(b)] = a[Number(b)] : this.o.set(b, a[b])) };
    l = ta.prototype;
    l.toString = function(a) { if (a && 0 <= a.indexOf(this)) return ""; for (var b = [], c = 0; c < this.g.length; c++) { var d = this.g[c];
            null === d || void 0 === d ? b.push("") : d instanceof ta ? (a = a || [], a.push(this), b.push(d.toString(a)), a.pop()) : b.push(d.toString()) } return b.join(",") };
    l.set = function(a, b) { if (!this.s)
            if ("length" === a) { if (!pa(b)) throw Error("RangeError: Length property must be a valid integer.");
                this.g.length = Number(b) } else pa(a) ? this.g[Number(a)] = b : this.o.set(a, b) };
    l.get = function(a) { return "length" === a ? this.length() : pa(a) ? this.g[Number(a)] : this.o.get(a) };
    l.length = function() { return this.g.length };
    l.Fb = function() { for (var a = ra(this.o, 1), b = 0; b < this.g.length; b++) a.push(b + ""); return new ta(a) };
    var ua = function(a, b) { pa(b) ? delete a.g[Number(b)] : sa(a.o, b) };
    l = ta.prototype;
    l.pop = function() { return this.g.pop() };
    l.push = function(a) { return this.g.push.apply(this.g, Array.prototype.slice.call(arguments)) };
    l.shift = function() { return this.g.shift() };
    l.splice = function(a, b, c) { return new ta(this.g.splice.apply(this.g, arguments)) };
    l.unshift = function(a) { return this.g.unshift.apply(this.g, Array.prototype.slice.call(arguments)) };
    l.has = function(a) { return pa(a) && this.g.hasOwnProperty(a) || this.o.has(a) };
    l.Gb = function() { this.s = !0;
        Object.freeze(this.g);
        this.o.Gb() };
    var wa = function() {
        function a(f, g) { if (b[f]) { if (b[f].Vd + g > b[f].max) throw Error("Quota exceeded");
                b[f].Vd += g } } var b = {},
            c = void 0,
            d = void 0,
            e = { Nj: function(f) { c = f }, Wg: function() { c && a(c, 1) }, Pj: function(f) { d = f }, Ib: function(f) { d && a(d, f) }, lk: function(f, g) { b[f] = b[f] || { Vd: 0 };
                    b[f].max = g }, oj: function(f) { return b[f] && b[f].Vd || 0 }, reset: function() { b = {} }, $i: a };
        e.onFnConsume = e.Nj;
        e.consumeFn = e.Wg;
        e.onStorageConsume = e.Pj;
        e.consumeStorage = e.Ib;
        e.setMax = e.lk;
        e.getConsumed = e.oj;
        e.reset = e.reset;
        e.consume = e.$i; return e };
    var xa = function(a, b) { this.s = a;
        this.P = function(c, d, e) { return c.apply(d, e) };
        this.C = b;
        this.o = new qa;
        this.g = this.I = void 0 };
    xa.prototype.add = function(a, b) { ya(this, a, b, !1) };
    var ya = function(a, b, c, d) { if (!a.o.s)
            if (a.s.Ib(("string" === typeof b ? b.length : 1) + ("string" === typeof c ? c.length : 1)), d) { var e = a.o;
                e.set(b, c);
                e.I["dust." + b] = !0 } else a.o.set(b, c) };
    xa.prototype.set = function(a, b) { this.o.s || (!this.o.has(a) && this.C && this.C.has(a) ? this.C.set(a, b) : (this.s.Ib(("string" === typeof a ? a.length : 1) + ("string" === typeof b ? b.length : 1)), this.o.set(a, b))) };
    xa.prototype.get = function(a) { return this.o.has(a) ? this.o.get(a) : this.C ? this.C.get(a) : void 0 };
    xa.prototype.has = function(a) { return !!this.o.has(a) || !(!this.C || !this.C.has(a)) };
    var za = function(a) { var b = new xa(a.s, a);
        a.I && (b.I = a.I);
        b.P = a.P;
        b.g = a.g; return b };
    var Aa = function() {},
        Ba = function(a) { return "function" === typeof a },
        Da = function(a) { return "string" === typeof a },
        Fa = function(a) { return "number" === typeof a && !isNaN(a) },
        Ha = Array.isArray,
        Ia = function(a, b) { if (a && Ha(a))
                for (var c = 0; c < a.length; c++)
                    if (a[c] && b(a[c])) return a[c] },
        Ja = function(a, b) { if (!Fa(a) || !Fa(b) || a > b) a = 0, b = 2147483647; return Math.floor(Math.random() * (b - a + 1) + a) },
        La = function(a, b) { for (var c = new Ka, d = 0; d < a.length; d++) c.set(a[d], !0); for (var e = 0; e < b.length; e++)
                if (c.get(b[e])) return !0;
            return !1 },
        Ma = function(a,
            b) { for (var c in a) Object.prototype.hasOwnProperty.call(a, c) && b(c, a[c]) },
        Na = function(a) { return !!a && ("[object Arguments]" === Object.prototype.toString.call(a) || Object.prototype.hasOwnProperty.call(a, "callee")) },
        Oa = function(a) { return Math.round(Number(a)) || 0 },
        Pa = function(a) { return "false" === String(a).toLowerCase() ? !1 : !!a },
        Qa = function(a) { var b = []; if (Ha(a))
                for (var c = 0; c < a.length; c++) b.push(String(a[c])); return b },
        Ra = function(a) { return a ? a.replace(/^\s+|\s+$/g, "") : "" },
        Sa = function() { return new Date(Date.now()) },
        Ua = function() { return Sa().getTime() },
        Ka = function() { this.prefix = "gtm.";
            this.values = {} };
    Ka.prototype.set = function(a, b) { this.values[this.prefix + a] = b };
    Ka.prototype.get = function(a) { return this.values[this.prefix + a] };
    var Va = function(a, b, c) { return a && a.hasOwnProperty(b) ? a[b] : c },
        Wa = function(a) { var b = a; return function() { if (b) { var c = b;
                    b = void 0; try { c() } catch (d) {} } } },
        Xa = function(a, b) { for (var c in b) b.hasOwnProperty(c) && (a[c] = b[c]) },
        Ya = function(a) { for (var b in a)
                if (a.hasOwnProperty(b)) return !0;
            return !1 },
        Za = function(a, b) { for (var c = [], d = 0; d < a.length; d++) c.push(a[d]), c.push.apply(c, b[a[d]] || []); return c },
        $a = function(a, b) {
            var c = m;
            b = b || [];
            for (var d = c, e = 0; e < a.length - 1; e++) {
                if (!d.hasOwnProperty(a[e])) return;
                d = d[a[e]];
                if (0 <=
                    b.indexOf(d)) return
            }
            return d
        },
        ab = function(a, b) { for (var c = {}, d = c, e = a.split("."), f = 0; f < e.length - 1; f++) d = d[e[f]] = {};
            d[e[e.length - 1]] = b; return c },
        bb = /^\w{1,9}$/,
        cb = function(a, b) { a = a || {};
            b = b || ","; var c = [];
            Ma(a, function(d, e) { bb.test(d) && e && c.push(d) }); return c.join(b) },
        db = function(a, b) {
            function c() {++d === b && (e(), e = null, c.done = !0) } var d = 0,
                e = a;
            c.done = !1; return c };
    var eb = function(a, b) { qa.call(this);
        this.P = a;
        this.ab = b };
    la(eb, qa);
    eb.prototype.toString = function() { return this.P };
    eb.prototype.Fb = function() { return new ta(ra(this, 1)) };
    eb.prototype.g = function(a, b) { a.s.Wg(); return this.ab.apply(new fb(this, a), Array.prototype.slice.call(arguments, 1)) };
    eb.prototype.o = function(a, b) { try { return this.g.apply(this, Array.prototype.slice.call(arguments, 0)) } catch (c) {} };
    var hb = function(a, b) { for (var c, d = 0; d < b.length && !(c = gb(a, b[d]), c instanceof oa); d++); return c },
        gb = function(a, b) { try { var c = a.get(String(b[0])); if (!(c && c instanceof eb)) throw Error("Attempting to execute non-function " + b[0] + "."); return c.g.apply(c, [a].concat(b.slice(1))) } catch (e) { var d = a.I;
                d && d(e, b.context ? { id: b[0], line: b.context.line } : null); throw e; } },
        fb = function(a, b) { this.o = a;
            this.g = b },
        z = function(a, b) { return Ha(b) ? gb(a.g, b) : b },
        G = function(a) { return a.o.P };
    var ib = function() { qa.call(this) };
    la(ib, qa);
    ib.prototype.Fb = function() { return new ta(ra(this, 1)) };
    var jb = {
        control: function(a, b) { return new oa(a, z(this, b)) },
        fn: function(a, b, c) {
            var d = this.g,
                e = z(this, b);
            if (!(e instanceof ta)) throw Error("Error: non-List value given for Fn argument names.");
            var f = Array.prototype.slice.call(arguments, 2);
            this.g.s.Ib(a.length + f.length);
            return new eb(a, function() {
                return function(g) {
                    var h = za(d);
                    void 0 === h.g && (h.g = this.g.g);
                    for (var k = Array.prototype.slice.call(arguments, 0), n = 0; n < k.length; n++)
                        if (k[n] = z(this, k[n]), k[n] instanceof oa) return k[n];
                    for (var p = e.get("length"), q =
                            0; q < p; q++) q < k.length ? h.add(e.get(q), k[q]) : h.add(e.get(q), void 0);
                    h.add("arguments", new ta(k));
                    var r = hb(h, f);
                    if (r instanceof oa) return "return" === r.g ? r.o : r
                }
            }())
        },
        list: function(a) { var b = this.g.s;
            b.Ib(arguments.length); for (var c = new ta, d = 0; d < arguments.length; d++) { var e = z(this, arguments[d]); "string" === typeof e && b.Ib(e.length ? e.length - 1 : 0);
                c.push(e) } return c },
        map: function(a) {
            for (var b = this.g.s, c = new ib, d = 0; d < arguments.length - 1; d += 2) {
                var e = z(this, arguments[d]) + "",
                    f = z(this, arguments[d + 1]),
                    g = e.length;
                g += "string" ===
                    typeof f ? f.length : 1;
                b.Ib(g);
                c.set(e, f)
            }
            return c
        },
        undefined: function() {}
    };
    var kb = function() { this.s = wa();
            this.g = new xa(this.s) },
        lb = function(a, b, c) { var d = new eb(b, c);
            d.Gb();
            a.g.set(b, d) },
        mb = function(a, b, c) { jb.hasOwnProperty(b) && lb(a, c || b, jb[b]) };
    kb.prototype.execute = function(a, b) { var c = Array.prototype.slice.call(arguments, 0); return this.o(c) };
    kb.prototype.o = function(a) { for (var b, c = 0; c < arguments.length; c++) b = gb(this.g, arguments[c]); return b };
    kb.prototype.C = function(a, b) { var c = za(this.g);
        c.g = a; for (var d, e = 1; e < arguments.length; e++) d = d = gb(c, arguments[e]); return d };
    var nb, ob = function() { if (void 0 === nb) { var a = null,
                b = ma.trustedTypes; if (b && b.createPolicy) { try { a = b.createPolicy("goog#html", { createHTML: na, createScript: na, createScriptURL: na }) } catch (c) { ma.console && ma.console.error(c.message) }
                nb = a } else nb = a } return nb };
    var qb = function(a, b) { this.g = b === pb ? a : "" };
    qb.prototype.toString = function() { return this.g + "" };
    var rb = function(a) { return a instanceof qb && a.constructor === qb ? a.g : "type_error:TrustedResourceUrl" },
        pb = {},
        sb = function(a) { var b = ob(),
                c = b ? b.createScriptURL(a) : a; return new qb(c, pb) };
    var tb = /^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i;

    function ub() { var a = ma.navigator; if (a) { var b = a.userAgent; if (b) return b } return "" }

    function vb(a) { return -1 != ub().indexOf(a) };
    var wb = {},
        xb = function(a, b, c) { this.g = c === wb ? a : "" };
    xb.prototype.toString = function() { return this.g.toString() };
    var yb = function(a) { return a instanceof xb && a.constructor === xb ? a.g : "type_error:SafeHtml" },
        zb = function(a) { var b = ob(),
                c = b ? b.createHTML(a) : a; return new xb(c, null, wb) };
    /*
        
         SPDX-License-Identifier: Apache-2.0
        */
    var Ab = {};
    var Bb = function() {},
        Cb = function(a) { this.g = a };
    la(Cb, Bb);
    Cb.prototype.toString = function() { return this.g };

    function Eb(a, b) { var c = [new Cb(Fb[0].toLowerCase(), Ab)]; if (0 === c.length) throw Error("No prefixes are provided"); var d = c.map(function(f) { var g; if (f instanceof Cb) g = f.g;
                else throw Error(""); return g }),
            e = b.toLowerCase(); if (d.every(function(f) { return 0 !== e.indexOf(f) })) throw Error('Attribute "' + b + '" does not match any of the allowed prefixes.');
        a.setAttribute(b, "true") }

    function Gb(a) { if ("script" === a.tagName.toLowerCase()) throw Error("Use setTextContent with a SafeScript."); if ("style" === a.tagName.toLowerCase()) throw Error("Use setTextContent with a SafeStyleSheet."); };
    var m = window,
        H = document,
        Hb = navigator,
        Ib = H.currentScript && H.currentScript.src,
        Jb = function(a, b) { var c = m[a];
            m[a] = void 0 === c ? b : c; return m[a] },
        Kb = function(a, b) { b && (a.addEventListener ? a.onload = b : a.onreadystatechange = function() { a.readyState in { loaded: 1, complete: 1 } && (a.onreadystatechange = null, b()) }) },
        Lb = { async: 1, nonce: 1, onerror: 1, onload: 1, src: 1, type: 1 },
        Mb = { onload: 1, src: 1, width: 1, height: 1, style: 1 };

    function Nb(a, b, c) { b && Ma(b, function(d, e) { d = d.toLowerCase();
            c.hasOwnProperty(d) || a.setAttribute(d, e) }) }
    var Ob = function(a, b, c, d) { var e = H.createElement("script");
            Nb(e, d, Lb);
            e.type = "text/javascript";
            e.async = !0; var f = sb(a);
            e.src = rb(f); var g, h, k = (e.ownerDocument && e.ownerDocument.defaultView || window).document,
                n = null === (h = k.querySelector) || void 0 === h ? void 0 : h.call(k, "script[nonce]");
            (g = n ? n.nonce || n.getAttribute("nonce") || "" : "") && e.setAttribute("nonce", g);
            Kb(e, b);
            c && (e.onerror = c); var p = H.getElementsByTagName("script")[0] || H.body || H.head;
            p.parentNode.insertBefore(e, p); return e },
        Pb = function() {
            if (Ib) {
                var a = Ib.toLowerCase();
                if (0 === a.indexOf("https://")) return 2;
                if (0 === a.indexOf("http://")) return 3
            }
            return 1
        },
        Qb = function(a, b, c, d, e) { var f = e,
                g = !1;
            f || (f = H.createElement("iframe"), g = !0);
            Nb(f, c, Mb);
            d && Ma(d, function(k, n) { f.dataset[k] = n });
            f.height = "0";
            f.width = "0";
            f.style.display = "none";
            f.style.visibility = "hidden"; if (g) { var h = H.body && H.body.lastChild || H.body || H.head;
                h.parentNode.insertBefore(f, h) }
            Kb(f, b);
            void 0 !== a && (f.src = a); return f },
        Rb = function(a, b, c) {
            var d = new Image(1, 1);
            d.onload = function() { d.onload = null;
                b && b() };
            d.onerror =
                function() { d.onerror = null;
                    c && c() };
            d.src = a;
            return d
        },
        Sb = function(a, b, c, d) { a.addEventListener ? a.addEventListener(b, c, !!d) : a.attachEvent && a.attachEvent("on" + b, c) },
        Tb = function(a, b, c) { a.removeEventListener ? a.removeEventListener(b, c, !1) : a.detachEvent && a.detachEvent("on" + b, c) },
        I = function(a) { m.setTimeout(a, 0) },
        Ub = function(a, b) { return a && b && a.attributes && a.attributes[b] ? a.attributes[b].value : null },
        Vb = function(a) {
            var b = a.innerText || a.textContent || "";
            b && " " != b && (b = b.replace(/^[\s\xa0]+|[\s\xa0]+$/g, ""));
            b &&
                (b = b.replace(/(\xa0+|\s{2,}|\n|\r\t)/g, " "));
            return b
        },
        Wb = function(a) { var b = H.createElement("div"),
                c = b,
                d = zb("A<div>" + a + "</div>");
            void 0 !== c.tagName && Gb(c);
            c.innerHTML = yb(d);
            b = b.lastChild; for (var e = []; b.firstChild;) e.push(b.removeChild(b.firstChild)); return e },
        Xb = function(a, b, c) { c = c || 100; for (var d = {}, e = 0; e < b.length; e++) d[b[e]] = !0; for (var f = a, g = 0; f && g <= c; g++) { if (d[String(f.tagName).toLowerCase()]) return f;
                f = f.parentElement } return null },
        Yb = function(a) {
            var b;
            try { b = Hb.sendBeacon && Hb.sendBeacon(a) } catch (c) {}
            b ||
                Rb(a)
        },
        Zb = function(a, b) { var c = a[b];
            c && "string" === typeof c.animVal && (c = c.animVal); return c },
        $b = function(a) { var b = H.featurePolicy; return b && Ba(b.allowsFeature) ? b.allowsFeature(a) : !1 };
    var ac = function(a, b) { return z(this, a) && z(this, b) },
        bc = function(a, b) { return z(this, a) === z(this, b) },
        cc = function(a, b) { return z(this, a) || z(this, b) },
        dc = function(a, b) { a = z(this, a);
            b = z(this, b); return -1 < String(a).indexOf(String(b)) },
        ec = function(a, b) { a = String(z(this, a));
            b = String(z(this, b)); return a.substring(0, b.length) === b },
        fc = function(a, b) { a = z(this, a);
            b = z(this, b); switch (a) {
                case "pageLocation":
                    var c = m.location.href;
                    b instanceof ib && b.get("stripProtocol") && (c = c.replace(/^https?:\/\//, "")); return c } };
    var hc = function() { this.g = new kb;
        gc(this) };
    hc.prototype.execute = function(a) { return this.g.o(a) };
    var gc = function(a) { mb(a.g, "map"); var b = function(c, d) { lb(a.g, c, d) };
        b("and", ac);
        b("contains", dc);
        b("equals", bc);
        b("or", cc);
        b("startsWith", ec);
        b("variable", fc) };
    var ic = function(a) { if (a instanceof ic) return a;
        this.hb = a };
    ic.prototype.toString = function() { return String(this.hb) };
    var kc = function(a) { qa.call(this);
        this.g = a;
        this.set("then", jc(this));
        this.set("catch", jc(this, !0));
        this.set("finally", jc(this, !1, !0)) };
    la(kc, ib);
    var jc = function(a, b, c) { b = void 0 === b ? !1 : b;
        c = void 0 === c ? !1 : c; return new eb("", function(d, e) { b && (e = d, d = void 0);
            c && (e = d);
            d instanceof eb || (d = void 0);
            e instanceof eb || (e = void 0); var f = za(this.g),
                g = function(k) { return function(n) { return c ? (k.g(f), a.g) : k.g(f, n) } },
                h = a.g.then(d && g(d), e && g(e)); return new kc(h) }) };
    /*
         jQuery (c) 2005, 2012 jQuery Foundation, Inc. jquery.org/license. */
    var lc = /\[object (Boolean|Number|String|Function|Array|Date|RegExp)\]/,
        mc = function(a) { if (null == a) return String(a); var b = lc.exec(Object.prototype.toString.call(Object(a))); return b ? b[1].toLowerCase() : "object" },
        nc = function(a, b) { return Object.prototype.hasOwnProperty.call(Object(a), b) },
        oc = function(a) {
            if (!a || "object" != mc(a) || a.nodeType || a == a.window) return !1;
            try { if (a.constructor && !nc(a, "constructor") && !nc(a.constructor.prototype, "isPrototypeOf")) return !1 } catch (c) { return !1 }
            for (var b in a);
            return void 0 ===
                b || nc(a, b)
        },
        pc = function(a, b) { var c = b || ("array" == mc(a) ? [] : {}),
                d; for (d in a)
                if (nc(a, d)) { var e = a[d]; "array" == mc(e) ? ("array" != mc(c[d]) && (c[d] = []), c[d] = pc(e, c[d])) : oc(e) ? (oc(c[d]) || (c[d] = {}), c[d] = pc(e, c[d])) : c[d] = e }
            return c };
    var rc = function(a, b, c) {
            var d = [],
                e = [],
                f = function(h, k) { for (var n = ra(h, 1), p = 0; p < n.length; p++) k[n[p]] = g(h.get(n[p])) },
                g = function(h) {
                    var k = d.indexOf(h);
                    if (-1 < k) return e[k];
                    if (h instanceof ta) { var n = [];
                        d.push(h);
                        e.push(n); for (var p = h.Fb(), q = 0; q < p.length(); q++) n[p.get(q)] = g(h.get(p.get(q))); return n }
                    if (h instanceof kc) return h.g;
                    if (h instanceof ib) { var r = {};
                        d.push(h);
                        e.push(r);
                        f(h, r); return r }
                    if (h instanceof eb) {
                        var u = function() {
                            for (var t = Array.prototype.slice.call(arguments, 0), v = 0; v < t.length; v++) t[v] = qc(t[v],
                                b, !!c);
                            var x = b ? b.s : wa(),
                                y = new xa(x);
                            b && (y.g = b.g);
                            return g(h.g.apply(h, [y].concat(t)))
                        };
                        d.push(h);
                        e.push(u);
                        f(h, u);
                        return u
                    }
                    switch (typeof h) {
                        case "boolean":
                        case "number":
                        case "string":
                        case "undefined":
                            return h;
                        case "object":
                            if (null === h) return null }
                };
            return g(a)
        },
        qc = function(a, b, c) {
            var d = [],
                e = [],
                f = function(h, k) {
                    for (var n in h) h.hasOwnProperty(n) && k.set(n,
                        g(h[n]))
                },
                g = function(h) { var k = d.indexOf(h); if (-1 < k) return e[k]; if (Ha(h) || Na(h)) { var n = new ta([]);
                        d.push(h);
                        e.push(n); for (var p in h) h.hasOwnProperty(p) && n.set(p, g(h[p])); return n } if (oc(h)) { var q = new ib;
                        d.push(h);
                        e.push(q);
                        f(h, q); return q } if ("function" === typeof h) { var r = new eb("", function(y) { for (var w = Array.prototype.slice.call(arguments, 0), A = 0; A < w.length; A++) w[A] = rc(z(this, w[A]), b, !!c); return g((0, this.g.P)(h, h, w)) });
                        d.push(h);
                        e.push(r);
                        f(h, r); return r } var x = typeof h; if (null === h || "string" === x || "number" === x || "boolean" === x) return h; };
            return g(a)
        };
    var sc = function(a) { for (var b = [], c = 0; c < a.length(); c++) a.has(c) && (b[c] = a.get(c)); return b },
        tc = function(a) { if (void 0 === a || Ha(a) || oc(a)) return !0; switch (typeof a) {
                case "boolean":
                case "number":
                case "string":
                case "function":
                    return !0 } return !1 };
    var uc = {
        supportedMethods: "concat every filter forEach hasOwnProperty indexOf join lastIndexOf map pop push reduce reduceRight reverse shift slice some sort splice unshift toString".split(" "),
        concat: function(a, b) { for (var c = [], d = 0; d < this.length(); d++) c.push(this.get(d)); for (var e = 1; e < arguments.length; e++)
                if (arguments[e] instanceof ta)
                    for (var f = arguments[e], g = 0; g < f.length(); g++) c.push(f.get(g));
                else c.push(arguments[e]);
            return new ta(c) },
        every: function(a, b) {
            for (var c = this.length(), d = 0; d < this.length() &&
                d < c; d++)
                if (this.has(d) && !b.g(a, this.get(d), d, this)) return !1;
            return !0
        },
        filter: function(a, b) { for (var c = this.length(), d = [], e = 0; e < this.length() && e < c; e++) this.has(e) && b.g(a, this.get(e), e, this) && d.push(this.get(e)); return new ta(d) },
        forEach: function(a, b) { for (var c = this.length(), d = 0; d < this.length() && d < c; d++) this.has(d) && b.g(a, this.get(d), d, this) },
        hasOwnProperty: function(a, b) { return this.has(b) },
        indexOf: function(a, b, c) {
            var d = this.length(),
                e = void 0 === c ? 0 : Number(c);
            0 > e && (e = Math.max(d + e, 0));
            for (var f = e; f < d; f++)
                if (this.has(f) &&
                    this.get(f) === b) return f;
            return -1
        },
        join: function(a, b) { for (var c = [], d = 0; d < this.length(); d++) c.push(this.get(d)); return c.join(b) },
        lastIndexOf: function(a, b, c) { var d = this.length(),
                e = d - 1;
            void 0 !== c && (e = 0 > c ? d + c : Math.min(c, e)); for (var f = e; 0 <= f; f--)
                if (this.has(f) && this.get(f) === b) return f;
            return -1 },
        map: function(a, b) { for (var c = this.length(), d = [], e = 0; e < this.length() && e < c; e++) this.has(e) && (d[e] = b.g(a, this.get(e), e, this)); return new ta(d) },
        pop: function() { return this.pop() },
        push: function(a, b) {
            return this.push.apply(this,
                Array.prototype.slice.call(arguments, 1))
        },
        reduce: function(a, b, c) { var d = this.length(),
                e, f = 0; if (void 0 !== c) e = c;
            else { if (0 === d) throw Error("TypeError: Reduce on List with no elements."); for (var g = 0; g < d; g++)
                    if (this.has(g)) { e = this.get(g);
                        f = g + 1; break }
                if (g === d) throw Error("TypeError: Reduce on List with no elements."); } for (var h = f; h < d; h++) this.has(h) && (e = b.g(a, e, this.get(h), h, this)); return e },
        reduceRight: function(a, b, c) {
            var d = this.length(),
                e, f = d - 1;
            if (void 0 !== c) e = c;
            else {
                if (0 === d) throw Error("TypeError: ReduceRight on List with no elements.");
                for (var g = 1; g <= d; g++)
                    if (this.has(d - g)) { e = this.get(d - g);
                        f = d - (g + 1); break }
                if (g > d) throw Error("TypeError: ReduceRight on List with no elements.");
            }
            for (var h = f; 0 <= h; h--) this.has(h) && (e = b.g(a, e, this.get(h), h, this));
            return e
        },
        reverse: function() { for (var a = sc(this), b = a.length - 1, c = 0; 0 <= b; b--, c++) a.hasOwnProperty(b) ? this.set(c, a[b]) : ua(this, c); return this },
        shift: function() { return this.shift() },
        slice: function(a, b, c) {
            var d = this.length();
            void 0 === b && (b = 0);
            b = 0 > b ? Math.max(d + b, 0) : Math.min(b, d);
            c = void 0 === c ? d : 0 > c ?
                Math.max(d + c, 0) : Math.min(c, d);
            c = Math.max(b, c);
            for (var e = [], f = b; f < c; f++) e.push(this.get(f));
            return new ta(e)
        },
        some: function(a, b) { for (var c = this.length(), d = 0; d < this.length() && d < c; d++)
                if (this.has(d) && b.g(a, this.get(d), d, this)) return !0;
            return !1 },
        sort: function(a, b) { var c = sc(this);
            void 0 === b ? c.sort() : c.sort(function(e, f) { return Number(b.g(a, e, f)) }); for (var d = 0; d < c.length; d++) c.hasOwnProperty(d) ? this.set(d, c[d]) : ua(this, d); return this },
        splice: function(a, b, c, d) {
            return this.splice.apply(this, Array.prototype.splice.call(arguments,
                1, arguments.length - 1))
        },
        toString: function() { return this.toString() },
        unshift: function(a, b) { return this.unshift.apply(this, Array.prototype.slice.call(arguments, 1)) }
    };
    var vc = "charAt concat indexOf lastIndexOf match replace search slice split substring toLowerCase toLocaleLowerCase toString toUpperCase toLocaleUpperCase trim".split(" "),
        wc = new oa("break"),
        xc = new oa("continue"),
        yc = function(a, b) { return z(this, a) + z(this, b) },
        Ac = function(a, b) { return z(this, a) && z(this, b) },
        Bc = function(a, b, c) {
            a = z(this, a);
            b = z(this, b);
            c = z(this, c);
            if (!(c instanceof ta)) throw Error("Error: Non-List argument given to Apply instruction.");
            if (null === a || void 0 === a) throw Error("TypeError: Can't read property " +
                b + " of " + a + ".");
            var d = "number" === typeof a;
            if ("boolean" === typeof a || d) { if ("toString" === b) { if (d && c.length()) { var e = rc(c.get(0)); try { return a.toString(e) } catch (q) {} } return a.toString() } throw Error("TypeError: " + a + "." + b + " is not a function."); }
            if ("string" === typeof a) { if (0 <= vc.indexOf(b)) { var f = rc(c); return qc(a[b].apply(a, f), this.g) } throw Error("TypeError: " + b + " is not a function"); }
            if (a instanceof ta) {
                if (a.has(b)) {
                    var g = a.get(b);
                    if (g instanceof eb) { var h = sc(c);
                        h.unshift(this.g); return g.g.apply(g, h) }
                    throw Error("TypeError: " +
                        b + " is not a function");
                }
                if (0 <= uc.supportedMethods.indexOf(b)) { var k = sc(c);
                    k.unshift(this.g); return uc[b].apply(a, k) }
            }
            if (a instanceof eb || a instanceof ib) { if (a.has(b)) { var n = a.get(b); if (n instanceof eb) { var p = sc(c);
                        p.unshift(this.g); return n.g.apply(n, p) } throw Error("TypeError: " + b + " is not a function"); } if ("toString" === b) return a instanceof eb ? a.P : a.toString(); if ("hasOwnProperty" === b) return a.has.apply(a, sc(c)) }
            if (a instanceof ic && "toString" === b) return a.toString();
            throw Error("TypeError: Object has no '" +
                b + "' property.");
        },
        Cc = function(a, b) { a = z(this, a); if ("string" !== typeof a) throw Error("Invalid key name given for assignment."); var c = this.g; if (!c.has(a)) throw Error("Attempting to assign to undefined value " + b); var d = z(this, b);
            c.set(a, d); return d },
        Dc = function(a) { var b = za(this.g),
                c = hb(b, Array.prototype.slice.apply(arguments)); if (c instanceof oa) return c },
        Ec = function() { return wc },
        Fc = function(a) { for (var b = z(this, a), c = 0; c < b.length; c++) { var d = z(this, b[c]); if (d instanceof oa) return d } },
        Gc = function(a) {
            for (var b =
                    this.g, c = 0; c < arguments.length - 1; c += 2) { var d = arguments[c]; if ("string" === typeof d) { var e = z(this, arguments[c + 1]);
                    ya(b, d, e, !0) } }
        },
        Hc = function() { return xc },
        Ic = function(a, b, c) { var d = new ta;
            b = z(this, b); for (var e = 0; e < b.length; e++) d.push(b[e]); var f = [51, a, d].concat(Array.prototype.splice.call(arguments, 2, arguments.length - 2));
            this.g.add(a, z(this, f)) },
        Jc = function(a, b) { return z(this, a) / z(this, b) },
        Kc = function(a, b) {
            a = z(this, a);
            b = z(this, b);
            var c = a instanceof ic,
                d = b instanceof ic;
            return c || d ? c && d ? a.hb == b.hb : !1 : a ==
                b
        },
        Lc = function(a) { for (var b, c = 0; c < arguments.length; c++) b = z(this, arguments[c]); return b };

    function Mc(a, b, c, d) { for (var e = 0; e < b(); e++) { var f = a(c(e)),
                g = hb(f, d); if (g instanceof oa) { if ("break" === g.g) break; if ("return" === g.g) return g } } }

    function Nc(a, b, c) { if ("string" === typeof b) return Mc(a, function() { return b.length }, function(f) { return f }, c); if (b instanceof ib || b instanceof ta || b instanceof eb) { var d = b.Fb(),
                e = d.length(); return Mc(a, function() { return e }, function(f) { return d.get(f) }, c) } }
    var Oc = function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Nc(function(e) { d.set(a, e); return d }, b, c) },
        Pc = function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Nc(function(e) { var f = za(d);
                ya(f, a, e, !0); return f }, b, c) },
        Qc = function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Nc(function(e) { var f = za(d);
                f.add(a, e); return f }, b, c) },
        Sc = function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Rc(function(e) { d.set(a, e); return d }, b, c) },
        Vc =
        function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Rc(function(e) { var f = za(d);
                ya(f, a, e, !0); return f }, b, c) },
        Wc = function(a, b, c) { a = z(this, a);
            b = z(this, b);
            c = z(this, c); var d = this.g; return Rc(function(e) { var f = za(d);
                f.add(a, e); return f }, b, c) };

    function Rc(a, b, c) { if ("string" === typeof b) return Mc(a, function() { return b.length }, function(d) { return b[d] }, c); if (b instanceof ta) return Mc(a, function() { return b.length() }, function(d) { return b.get(d) }, c); throw new TypeError("The value is not iterable."); }
    var Xc = function(a, b, c, d) {
            function e(p, q) { for (var r = 0; r < f.length(); r++) { var u = f.get(r);
                    q.add(u, p.get(u)) } } var f = z(this, a); if (!(f instanceof ta)) throw Error("TypeError: Non-List argument given to ForLet instruction."); var g = this.g;
            d = z(this, d); var h = za(g); for (e(g, h); gb(h, b);) { var k = hb(h, d); if (k instanceof oa) { if ("break" === k.g) break; if ("return" === k.g) return k } var n = za(g);
                e(h, n);
                gb(n, c);
                h = n } },
        Yc = function(a) { a = z(this, a); var b = this.g,
                c = !1; if (c && !b.has(a)) throw new ReferenceError(a + " is not defined."); return b.get(a) },
        Zc = function(a, b) { var c;
            a = z(this, a);
            b = z(this, b); if (void 0 === a || null === a) throw Error("TypeError: cannot access property of " + a + "."); if (a instanceof ib || a instanceof ta || a instanceof eb) c = a.get(b);
            else if ("string" === typeof a) "length" === b ? c = a.length : pa(b) && (c = a[b]);
            else if (a instanceof ic) return; return c },
        $c = function(a, b) {
            return z(this, a) > z(this,
                b)
        },
        ad = function(a, b) { return z(this, a) >= z(this, b) },
        bd = function(a, b) { a = z(this, a);
            b = z(this, b);
            a instanceof ic && (a = a.hb);
            b instanceof ic && (b = b.hb); return a === b },
        cd = function(a, b) { return !bd.call(this, a, b) },
        dd = function(a, b, c) { var d = [];
            z(this, a) ? d = z(this, b) : c && (d = z(this, c)); var e = hb(this.g, d); if (e instanceof oa) return e },
        ed = function(a, b) { return z(this, a) < z(this, b) },
        fd = function(a, b) { return z(this, a) <= z(this, b) },
        gd = function(a, b) { return z(this, a) % z(this, b) },
        hd = function(a, b) { return z(this, a) * z(this, b) },
        id = function(a) {
            return -z(this,
                a)
        },
        jd = function(a) { return !z(this, a) },
        kd = function(a, b) { return !Kc.call(this, a, b) },
        ld = function() { return null },
        md = function(a, b) { return z(this, a) || z(this, b) },
        nd = function(a, b) { var c = z(this, a);
            z(this, b); return c },
        od = function(a) { return z(this, a) },
        pd = function(a) { return Array.prototype.slice.apply(arguments) },
        sd = function(a) { return new oa("return", z(this, a)) },
        td = function(a, b, c) {
            a = z(this, a);
            b = z(this, b);
            c = z(this, c);
            if (null === a || void 0 === a) throw Error("TypeError: Can't set property " + b + " of " + a + ".");
            (a instanceof eb || a instanceof ta || a instanceof ib) && a.set(b, c);
            return c
        },
        ud = function(a, b) { return z(this, a) - z(this, b) },
        vd = function(a, b, c) {
            a = z(this, a);
            var d = z(this, b),
                e = z(this, c);
            if (!Ha(d) || !Ha(e)) throw Error("Error: Malformed switch instruction.");
            for (var f, g = !1, h = 0; h < d.length; h++)
                if (g || a === z(this, d[h]))
                    if (f = z(this, e[h]), f instanceof oa) { var k = f.g; if ("break" === k) return; if ("return" === k || "continue" === k) return f } else g = !0;
            if (e.length === d.length + 1 && (f = z(this, e[e.length - 1]), f instanceof oa && ("return" === f.g || "continue" ===
                    f.g))) return f
        },
        wd = function(a, b, c) { return z(this, a) ? z(this, b) : z(this, c) },
        xd = function(a) { a = z(this, a); return a instanceof eb ? "function" : typeof a },
        yd = function(a) { for (var b = this.g, c = 0; c < arguments.length; c++) { var d = arguments[c]; "string" !== typeof d || b.add(d, void 0) } },
        zd = function(a, b, c, d) {
            var e = z(this, d);
            if (z(this, c)) { var f = hb(this.g, e); if (f instanceof oa) { if ("break" === f.g) return; if ("return" === f.g) return f } }
            for (; z(this, a);) {
                var g = hb(this.g, e);
                if (g instanceof oa) { if ("break" === g.g) break; if ("return" === g.g) return g }
                z(this,
                    b)
            }
        },
        Ad = function(a) { return ~Number(z(this, a)) },
        Bd = function(a, b) { return Number(z(this, a)) << Number(z(this, b)) },
        Cd = function(a, b) { return Number(z(this, a)) >> Number(z(this, b)) },
        Dd = function(a, b) { return Number(z(this, a)) >>> Number(z(this, b)) },
        Ed = function(a, b) { return Number(z(this, a)) & Number(z(this, b)) },
        Fd = function(a, b) { return Number(z(this, a)) ^ Number(z(this, b)) },
        Gd = function(a, b) { return Number(z(this, a)) | Number(z(this, b)) };
    var Id = function() { this.g = new kb;
        Hd(this) };
    Id.prototype.execute = function(a) { return Jd(this.g.o(a)) };
    var Kd = function(a, b, c) { return Jd(a.g.C(b, c)) },
        Hd = function(a) {
            var b = function(d, e) { mb(a.g, d, String(e)) };
            b("control", 49);
            b("fn", 51);
            b("list", 7);
            b("map", 8);
            b("undefined", 44);
            var c = function(d, e) { lb(a.g, String(d), e) };
            c(0, yc);
            c(1, Ac);
            c(2, Bc);
            c(3, Cc);
            c(53, Dc);
            c(4, Ec);
            c(5, Fc);
            c(52, Gc);
            c(6, Hc);
            c(9, Fc);
            c(50, Ic);
            c(10, Jc);
            c(12, Kc);
            c(13, Lc);
            c(47, Oc);
            c(54, Pc);
            c(55, Qc);
            c(63, Xc);
            c(64, Sc);
            c(65, Vc);
            c(66, Wc);
            c(15, Yc);
            c(16, Zc);
            c(17, Zc);
            c(18, $c);
            c(19, ad);
            c(20, bd);
            c(21, cd);
            c(22, dd);
            c(23, ed);
            c(24, fd);
            c(25, gd);
            c(26, hd);
            c(27,
                id);
            c(28, jd);
            c(29, kd);
            c(45, ld);
            c(30, md);
            c(32, nd);
            c(33, nd);
            c(34, od);
            c(35, od);
            c(46, pd);
            c(36, sd);
            c(43, td);
            c(37, ud);
            c(38, vd);
            c(39, wd);
            c(40, xd);
            c(41, yd);
            c(42, zd);
            c(58, Ad);
            c(57, Bd);
            c(60, Cd);
            c(61, Dd);
            c(56, Ed);
            c(62, Fd);
            c(59, Gd)
        };

    function Jd(a) { if (a instanceof oa || a instanceof eb || a instanceof ta || a instanceof ib || a instanceof ic || null === a || void 0 === a || "string" === typeof a || "number" === typeof a || "boolean" === typeof a) return a };
    var Ld = function() {
        var a = function(b) { return { toString: function() { return b } } };
        return {
            Qh: a("consent"),
            se: a("consent_always_fire"),
            Uf: a("convert_case_to"),
            Vf: a("convert_false_to"),
            Wf: a("convert_null_to"),
            Xf: a("convert_true_to"),
            Yf: a("convert_undefined_to"),
            wk: a("debug_mode_metadata"),
            Eb: a("function"),
            Bi: a("instance_name"),
            Fi: a("live_only"),
            Gi: a("malware_disabled"),
            Hi: a("metadata"),
            Mi: a("original_activity_id"),
            Bk: a("original_vendor_template_id"),
            Ak: a("once_on_load"),
            Li: a("once_per_event"),
            Bg: a("once_per_load"),
            Dk: a("priority_override"),
            Ek: a("respected_consent_types"),
            Hg: a("setup_tags"),
            Jg: a("tag_id"),
            Kg: a("teardown_tags")
        }
    }();
    var he;
    var ie = [],
        je = [],
        ke = [],
        le = [],
        me = [],
        ne = {},
        oe, pe, qe, re = function(a, b) { var c = {};
            c["function"] = "__" + a; for (var d in b) b.hasOwnProperty(d) && (c["vtp_" + d] = b[d]); return c },
        se = function(a, b) { var c = a["function"],
                d = b && b.event; if (!c) throw Error("Error: No function name given for function call."); var e = ne[c],
                f = {},
                g; for (g in a)
                if (a.hasOwnProperty(g))
                    if (0 === g.indexOf("vtp_")) e && d && d.Vg && d.Vg(a[g]), f[void 0 !== e ? g : g.substr(4)] = a[g];
                    else if (g === Ld.se.toString() && a[g]) {}
            e && d && d.Ug && (f.vtp_gtmCachedValues = d.Ug); return void 0 !== e ? e(f) : he(c, f, b) },
        ue = function(a, b, c) { c = c || []; var d = {},
                e; for (e in a) a.hasOwnProperty(e) && (d[e] = te(a[e], b, c)); return d },
        te = function(a, b, c) {
            if (Ha(a)) {
                var d;
                switch (a[0]) {
                    case "function_id":
                        return a[1];
                    case "list":
                        d = [];
                        for (var e = 1; e < a.length; e++) d.push(te(a[e], b, c));
                        return d;
                    case "macro":
                        var f = a[1];
                        if (c[f]) return;
                        var g = ie[f];
                        if (!g || b.zf(g)) return;
                        c[f] = !0;
                        try {
                            var h = ue(g, b, c);
                            h.vtp_gtmEventId =
                                b.id;
                            d = se(h, { event: b, index: f, type: 2 });
                            qe && (d = qe.aj(d, h))
                        } catch (y) { b.ph && b.ph(y, Number(f)), d = !1 }
                        c[f] = !1;
                        return d;
                    case "map":
                        d = {};
                        for (var k = 1; k < a.length; k += 2) d[te(a[k], b, c)] = te(a[k + 1], b, c);
                        return d;
                    case "template":
                        d = [];
                        for (var n = !1, p = 1; p < a.length; p++) { var q = te(a[p], b, c);
                            pe && (n = n || q === pe.Ld);
                            d.push(q) }
                        return pe && n ? pe.ej(d) : d.join("");
                    case "escape":
                        d = te(a[1], b, c);
                        if (pe && Ha(a[1]) && "macro" === a[1][0] && pe.Cj(a)) return pe.Uj(d);
                        d = String(d);
                        for (var r = 2; r < a.length; r++) Md[a[r]] && (d = Md[a[r]](d));
                        return d;
                    case "tag":
                        var u =
                            a[1];
                        if (!le[u]) throw Error("Unable to resolve tag reference " + u + ".");
                        return d = { eh: a[2], index: u };
                    case "zb":
                        var t = { arg0: a[2], arg1: a[3], ignore_case: a[5] };
                        t["function"] = a[1];
                        var v = ve(t, b, c),
                            x = !!a[4];
                        return x || 2 !== v ? x !== (1 === v) : null;
                    default:
                        throw Error("Attempting to expand unknown Value type: " + a[0] + ".");
                }
            }
            return a
        },
        ve = function(a, b, c) { try { return oe(ue(a, b, c)) } catch (d) { JSON.stringify(a) } return 2 };
    var we = function(a, b, c) { var d;
        d = Error.call(this);
        this.message = d.message; "stack" in d && (this.stack = d.stack);
        this.o = a;
        this.g = c };
    la(we, Error);

    function xe(a, b) { if (Ha(a)) { Object.defineProperty(a, "context", { value: { line: b[0] } }); for (var c = 1; c < a.length; c++) xe(a[c], b[c]) } };
    var ye = function(a, b) { var c;
        c = Error.call(this);
        this.message = c.message; "stack" in c && (this.stack = c.stack);
        this.Qj = a;
        this.o = b;
        this.g = [] };
    la(ye, Error);
    var Ae = function() { return function(a, b) { a instanceof ye || (a = new ye(a, ze));
            b && a.g.push(b); throw a; } };

    function ze(a) { if (!a.length) return a;
        a.push({ id: "main", line: 0 }); for (var b = a.length - 1; 0 < b; b--) Fa(a[b].id) && a.splice(b++, 1); for (var c = a.length - 1; 0 < c; c--) a[c].line = a[c - 1].line;
        a.splice(0, 1); return a };
    var De = function(a) {
            function b(r) { for (var u = 0; u < r.length; u++) d[r[u]] = !0 } for (var c = [], d = [], e = Be(a), f = 0; f < je.length; f++) { var g = je[f],
                    h = Ce(g, e); if (h) { for (var k = g.add || [], n = 0; n < k.length; n++) c[k[n]] = !0;
                    b(g.block || []) } else null === h && b(g.block || []); } for (var p = [], q = 0; q < le.length; q++) c[q] && !d[q] && (p[q] = !0); return p },
        Ce = function(a, b) {
            for (var c = a["if"] || [], d = 0; d < c.length; d++) { var e = b(c[d]); if (0 === e) return !1; if (2 === e) return null }
            for (var f =
                    a.unless || [], g = 0; g < f.length; g++) { var h = b(f[g]); if (2 === h) return null; if (1 === h) return !1 }
            return !0
        },
        Be = function(a) { var b = []; return function(c) { void 0 === b[c] && (b[c] = ve(ke[c], a)); return b[c] } };
    var Ee = { aj: function(a, b) { b[Ld.Uf] && "string" === typeof a && (a = 1 == b[Ld.Uf] ? a.toLowerCase() : a.toUpperCase());
            b.hasOwnProperty(Ld.Wf) && null === a && (a = b[Ld.Wf]);
            b.hasOwnProperty(Ld.Yf) && void 0 === a && (a = b[Ld.Yf]);
            b.hasOwnProperty(Ld.Xf) && !0 === a && (a = b[Ld.Xf]);
            b.hasOwnProperty(Ld.Vf) && !1 === a && (a = b[Ld.Vf]); return a } };
    var Fe = function() { this.g = {} };

    function Ge(a, b, c, d) { if (a)
            for (var e = 0; e < a.length; e++) { var f = void 0,
                    g = "A policy function denied the permission request"; try { f = a[e].call(void 0, b, c, d), g += "." } catch (h) { g = "string" === typeof h ? g + (": " + h) : h instanceof Error ? g + (": " + h.message) : g + "." } if (!f) throw new we(c, d, g); } }

    function He(a, b, c) { return function() { var d = arguments[0]; if (d) { var e = a.g[d],
                    f = a.g.all; if (e || f) { var g = c.apply(void 0, Array.prototype.slice.call(arguments, 0));
                    Ge(e, b, d, g);
                    Ge(f, b, d, g) } } } };
    var Ke = function() { var a = data.permissions || {},
                b = J.J,
                c = this;
            this.o = new Fe;
            this.g = {}; var d = {},
                e = He(this.o, b, function() { var f = arguments[0]; return f && d[f] ? d[f].apply(void 0, Array.prototype.slice.call(arguments, 0)) : {} });
            Ma(a, function(f, g) { var h = {};
                Ma(g, function(k, n) { var p = Ie(k, n);
                    h[k] = p.assert;
                    d[k] || (d[k] = p.V) });
                c.g[f] = function(k, n) { var p = h[k]; if (!p) throw Je(k, {}, "The requested permission " + k + " is not configured."); var q = Array.prototype.slice.call(arguments, 0);
                    p.apply(void 0, q);
                    e.apply(void 0, q) } }) },
        Me =
        function(a) { return Le.g[a] || function() {} };

    function Ie(a, b) { var c = re(a, b);
        c.vtp_permissionName = a;
        c.vtp_createPermissionError = Je; try { return se(c) } catch (d) { return { assert: function(e) { throw new we(e, {}, "Permission " + e + " is unknown."); }, V: function() { for (var e = {}, f = 0; f < arguments.length; ++f) e["arg" + (f + 1)] = arguments[f]; return e } } } }

    function Je(a, b, c) { return new we(a, b, c) };
    var Ne = !1;
    var Oe = {};
    Oe.vk = Pa('');
    Oe.ij = Pa('');
    var Pe = Ne,
        Qe = Oe.ij,
        Re = Oe.vk;
    var Se = function(a, b) { var c = String(a); return c };
    var Xe = function(a) { var b = {},
                c = 0;
            Ma(a, function(e, f) { if (void 0 !== f)
                    if (f = Se(f, 100), Te.hasOwnProperty(e)) b[Te[e]] = Ue(f);
                    else if (Ve.hasOwnProperty(e)) { var g = Ve[e],
                        h = Ue(f);
                    b.hasOwnProperty(g) || (b[g] = h) } else if ("category" === e)
                    for (var k = Ue(f).split("/", 5), n = 0; n < k.length; n++) { var p = We[n],
                            q = k[n];
                        b.hasOwnProperty(p) || (b[p] = q) } else 10 > c && (b["k" + c] = Ue(Se(e, 40)), b["v" + c] = Ue(f), c++) }); var d = [];
            Ma(b, function(e, f) { d.push("" + e + f) }); return d.join("~") },
        Ue = function(a) { return ("" + a).replace(/~/g, function() { return "~~" }) },
        Te = { item_id: "id", item_name: "nm", item_brand: "br", item_category: "ca", item_category2: "c2", item_category3: "c3", item_category4: "c4", item_category5: "c5", item_variant: "va", price: "pr", quantity: "qt", coupon: "cp", item_list_name: "ln", index: "lp", item_list_id: "li", discount: "ds", affiliation: "af", promotion_id: "pi", promotion_name: "pn", creative_name: "cn", creative_slot: "cs", location_id: "lo" },
        Ve = { id: "id", name: "nm", brand: "br", variant: "va", list_name: "ln", list_position: "lp", list: "ln", position: "lp", creative: "cn" },
        We = ["ca",
            "c2", "c3", "c4", "c5"
        ];
    var Ye = function(a) { var b = [];
            Ma(a, function(c, d) { null != d && b.push(encodeURIComponent(c) + "=" + encodeURIComponent(String(d))) }); return b.join("&") },
        Ze = function(a, b, c, d) { this.Ea = a.Ea;
            this.Rb = a.Rb;
            this.M = a.M;
            this.o = b;
            this.C = c;
            this.s = Ye(a.Ea);
            this.g = Ye(a.M);
            this.I = this.g.length; if (d && 16384 < this.I) throw Error("EVENT_TOO_LARGE"); };
    var $e = function() { this.events = [];
        this.g = this.Ea = "";
        this.s = 0;
        this.o = !1 };
    $e.prototype.add = function(a) { return this.C(a) ? (this.events.push(a), this.Ea = a.s, this.g = a.o, this.s += a.I, this.o = a.C, !0) : !1 };
    $e.prototype.C = function(a) { var b = 20 > this.events.length && 16384 > a.I + this.s,
            c = this.Ea === a.s && this.g === a.o && this.o === a.C; return 0 == this.events.length || b && c };

    var af = function(a, b) { Ma(a, function(c, d) { null != d && b.push(encodeURIComponent(c) + "=" + encodeURIComponent(d)) }) },
        bf = function(a, b) { var c = [];
            a.s && c.push(a.s);
            b && c.push("_s=" + b);
            af(a.Rb, c); var d = !1;
            a.g && (c.push(a.g), d = !0); var e = c.join("&"),
                f = "",
                g = e.length + a.o.length + 1;
            d && 2048 < g && (f = c.pop(), e = c.join("&")); return { If: e, body: f } },
        cf = function(a, b) {
            var c = a.events;
            if (1 == c.length) return bf(c[0], b);
            var d = [];
            a.Ea && d.push(a.Ea);
            for (var e = {}, f = 0; f < c.length; f++) Ma(c[f].Rb, function(u, t) {
                null != t && (e[u] = e[u] || {}, e[u][String(t)] =
                    e[u][String(t)] + 1 || 1)
            });
            var g = {};
            Ma(e, function(u, t) { var v, x = -1,
                    y = 0;
                Ma(t, function(w, A) { y += A; var B = (w.length + u.length + 2) * (A - 1);
                    B > x && (v = w, x = B) });
                y == c.length && (g[u] = v) });
            af(g, d);
            b && d.push("_s=" + b);
            for (var h = d.join("&"), k = [], n = {}, p = 0; p < c.length; n = { vd: n.vd }, p++) { var q = [];
                n.vd = {};
                Ma(c[p].Rb, function(u) { return function(t, v) { g[t] != "" + v && (u.vd[t] = v) } }(n));
                c[p].g && q.push(c[p].g);
                af(n.vd, q);
                k.push(q.join("&")) }
            var r = k.join("\r\n");
            return { If: h, body: r }
        };
    var qf = /^([a-z][a-z0-9]*):(!|\?)(\*|string|boolean|number|Fn|DustMap|List|OpaqueValue)$/i,
        rf = { Fn: "function", DustMap: "Object", List: "Array" },
        M = function(a, b, c) {
            for (var d = 0; d < b.length; d++) {
                var e = qf.exec(b[d]);
                if (!e) throw Error("Internal Error in " + a);
                var f = e[1],
                    g = "!" === e[2],
                    h = e[3],
                    k = c[d];
                if (null == k) { if (g) throw Error("Error in " + a + ". Required argument " + f + " not supplied."); } else if ("*" !== h) {
                    var n = typeof k;
                    k instanceof eb ? n = "Fn" : k instanceof ta ? n = "List" : k instanceof ib ? n = "DustMap" : k instanceof ic && (n = "OpaqueValue");
                    if (n != h) throw Error("Error in " + a + ". Argument " + f + " has type " + (rf[n] || n) + ", which does not match required type " + (rf[h] || h) + ".");
                }
            }
        };

    function sf(a) { return "" + a }

    function tf(a, b) { var c = []; return c };
    var uf = function(a, b) { var c = new eb(a, function() { for (var d = Array.prototype.slice.call(arguments, 0), e = 0; e < d.length; e++) d[e] = z(this, d[e]); return b.apply(this, d) });
            c.Gb(); return c },
        vf = function(a, b) { var c = new ib,
                d; for (d in b)
                if (b.hasOwnProperty(d)) { var e = b[d];
                    Ba(e) ? c.set(d, uf(a + "_" + d, e)) : (Fa(e) || Da(e) || "boolean" === typeof e) && c.set(d, e) }
            c.Gb(); return c };
    var wf = function(a, b) { M(G(this), ["apiName:!string", "message:?string"], arguments); var c = {},
            d = new ib; return d = vf("AssertApiSubject", c) };
    var xf = function(a, b) { M(G(this), ["actual:?*", "message:?string"], arguments); if (a instanceof kc) throw Error("Argument actual cannot have type Promise. Assertions on asynchronous code aren't supported."); var c = {},
            d = new ib; return d = vf("AssertThatSubject", c) };

    function yf(a) { return function() { for (var b = [], c = this.g, d = 0; d < arguments.length; ++d) b.push(rc(arguments[d], c)); return qc(a.apply(null, b)) } }
    var Af = function() { for (var a = Math, b = zf, c = {}, d = 0; d < b.length; d++) { var e = b[d];
            a.hasOwnProperty(e) && (c[e] = yf(a[e].bind(a))) } return c };
    var Bf = function(a) { var b; return b };
    var Cf = function(a) { var b; return b };
    var Df = function(a) { return encodeURI(a) };
    var Ef = function(a) { return encodeURIComponent(a) };
    var Ff = function(a) { M(G(this), ["message:?string"], arguments); };
    var Gf = function(a, b) { M(G(this), ["min:!number", "max:!number"], arguments); return Ja(a, b) };
    var N = function(a, b, c) { var d = a.g.g; if (!d) throw Error("Missing program state.");
        d.Wi.apply(null, Array.prototype.slice.call(arguments, 1)) };
    var Hf = function() { N(this, "read_container_data"); var a = new ib;
        a.set("containerId", 'G-CD35DZJ728');
        a.set("version", '1');
        a.set("environmentName", '');
        a.set("debugMode", Pe);
        a.set("previewMode", Re);
        a.set("environmentMode", Qe);
        a.Gb(); return a };
    var If = {};
    If.sstECEnableData = !0;
    If.enableAdsEc = !0;
    If.sstFFConversionEnabled = !0;
    If.sstEnableAuid = !0;
    If.enableGbraidUpdate = !0;
    If.enable1pScripts = !0;
    If.enableGlobalEventDeveloperIds = !1;
    If.enableGa4OnoRemarketing = !1;
    If.omitAuidIfWbraidPresent = !1;
    If.sstEnableDclid = !0;
    If.reconcileCampaignFields = !1;
    If.enableEmFormCcd = !1;
    If.enableEmFormCcdPart2 = !1;
    If.enableUrlPassthrough = !0;
    If.enableLandingPageDeduplication = !0;
    If.requireGtagUserDataTos = !0;

    function Jf() { return qc(If) };
    var Kf = function() { return (new Date).getTime() };
    var Lf = function(a) { if (null === a) return "null"; if (a instanceof ta) return "array"; if (a instanceof eb) return "function"; if (a instanceof ic) { a = a.hb; if (void 0 === a.constructor || void 0 === a.constructor.name) { var b = String(a); return b.substring(8, b.length - 1) } return String(a.constructor.name) } return typeof a };
    var Mf = function(a) {
        function b(c) { return function(d) { try { return c(d) } catch (e) {
                    (Pe || Re) && a.call(this, e.message) } } } return { parse: b(function(c) { return qc(JSON.parse(c)) }), stringify: b(function(c) { return JSON.stringify(rc(c)) }) } };
    var Nf = function(a) { return Oa(rc(a, this.g)) };
    var Of = function(a) { return Number(rc(a, this.g)) };
    var Pf = function(a) { return null === a ? "null" : void 0 === a ? "undefined" : a.toString() };
    var Qf = function(a, b, c) { var d = null,
            e = !1; return e ? d : null };
    var zf = "floor ceil round max min abs pow sqrt".split(" ");
    var Rf = function() { var a = {}; return { qj: function(b) { return a.hasOwnProperty(b) ? a[b] : void 0 }, mk: function(b, c) { a[b] = c }, reset: function() { a = {} } } },
        Sf = function(a, b) { return function() { var c = Array.prototype.slice.call(arguments, 0);
                c.unshift(b); return eb.prototype.g.apply(a, c) } },
        Tf = function(a, b) { M(G(this), ["apiName:!string", "mock:?*"], arguments); };
    var Uf = {};
    Uf.keys = function(a) { return new ta };
    Uf.values = function(a) { return new ta };
    Uf.entries = function(a) { return new ta };
    Uf.freeze = function(a) { return a };
    Uf.delete = function(a, b) { return !1 };
    var Wf = function() { this.g = {};
        this.o = {}; };
    Wf.prototype.get = function(a, b) { var c = this.g.hasOwnProperty(a) ? this.g[a] : void 0; return c };
    Wf.prototype.add = function(a, b, c) { if (this.g.hasOwnProperty(a)) throw "Attempting to add a function which already exists: " + a + "."; if (this.o.hasOwnProperty(a)) throw "Attempting to add an API with an existing private API name: " + a + ".";
        this.g[a] = c ? void 0 : Ba(b) ? uf(a, b) : vf(a, b) };
    var Yf = function(a, b, c) { if (a.o.hasOwnProperty(b)) throw "Attempting to add a private function which already exists: " + b + "."; if (a.g.hasOwnProperty(b)) throw "Attempting to add a private function with an existing API name: " + b + ".";
        a.o[b] = Ba(c) ? uf(b, c) : vf(b, c) };

    function Xf(a, b) { var c = void 0; return c };

    function Zf() { var a = {}; return a };
    var S = {
        Db: "_ee",
        mc: "_syn_or_mod",
        Fk: "_uei",
        Tc: "_eu",
        Ck: "_pci",
        bc: "event_callback",
        Dd: "event_timeout",
        Ga: "gtag.config",
        Pa: "gtag.get",
        va: "purchase",
        Zb: "refund",
        yb: "begin_checkout",
        Wb: "add_to_cart",
        Xb: "remove_from_cart",
        Zh: "view_cart",
        $f: "add_to_wishlist",
        wa: "view_item",
        Yb: "view_promotion",
        xe: "select_promotion",
        we: "select_item",
        zb: "view_item_list",
        Zf: "add_payment_info",
        Yh: "add_shipping_info",
        eb: "value_key",
        pb: "value_callback",
        aa: "allow_ad_personalization_signals",
        Pc: "restricted_data_processing",
        Bc: "allow_google_signals",
        za: "cookie_expires",
        ac: "cookie_update",
        Qc: "session_duration",
        Hd: "session_engaged_time",
        Bd: "engagement_time_msec",
        Ka: "user_properties",
        Ba: "transport_url",
        fa: "ads_data_redaction",
        Ca: "user_data",
        Kc: "first_party_collection",
        D: "ad_storage",
        R: "analytics_storage",
        te: "region",
        Tf: "wait_for_update",
        ya: "conversion_linker",
        Qa: "conversion_cookie_prefix",
        la: "value",
        ja: "currency",
        vg: "trip_type",
        ba: "items",
        lg: "passengers",
        Ae: "allow_custom_scripts",
        Cb: "session_id",
        qg: "quantity",
        sb: "transaction_id",
        rb: "language",
        Ad: "country",
        zd: "allow_enhanced_conversions",
        Fe: "aw_merchant_id",
        De: "aw_feed_country",
        Ee: "aw_feed_language",
        Ce: "discount",
        X: "developer_id",
        ig: "global_developer_id_string",
        fg: "event_developer_id_string",
        Id: "delivery_postal_code",
        Le: "estimated_delivery_date",
        Je: "shipping",
        Pe: "new_customer",
        Ge: "customer_lifetime_value",
        Ke: "enhanced_conversions",
        Ac: "page_view",
        oa: "linker",
        T: "domains",
        fc: "decorate_forms",
        eg: "enhanced_conversions_automatic_settings",
        fi: "auto_detection_enabled",
        gg: "ga_temp_client_id",
        ye: "user_engagement",
        Th: "app_remove",
        Uh: "app_store_refund",
        Vh: "app_store_subscription_cancel",
        Wh: "app_store_subscription_convert",
        Xh: "app_store_subscription_renew",
        $h: "first_open",
        ai: "first_visit",
        bi: "in_app_purchase",
        ci: "session_start",
        di: "allow_display_features",
        cb: "campaign",
        Cc: "campaign_content",
        Dc: "campaign_id",
        Ec: "campaign_medium",
        Fc: "campaign_name",
        Gc: "campaign_source",
        Hc: "campaign_term",
        Ha: "client_id",
        na: "cookie_domain",
        $b: "cookie_name",
        nb: "cookie_path",
        Ra: "cookie_flags",
        Ic: "custom_map",
        Ne: "groups",
        kg: "non_interaction",
        fb: "page_location",
        Qe: "page_path",
        Sa: "page_referrer",
        Oc: "page_title",
        Aa: "send_page_view",
        hc: "send_to",
        Rc: "session_engaged",
        Jc: "euid_logged_in_state",
        Sc: "session_number",
        yi: "tracking_id",
        tb: "url_passthrough",
        cc: "accept_incoming",
        Nc: "url_position",
        og: "phone_conversion_number",
        mg: "phone_conversion_callback",
        ng: "phone_conversion_css_class",
        pg: "phone_conversion_options",
        si: "phone_conversion_ids",
        ri: "phone_conversion_country_code",
        Ab: "aw_remarketing",
        Be: "aw_remarketing_only",
        ze: "gclid",
        ei: "auid",
        ki: "affiliation",
        dg: "tax",
        Ie: "list_name",
        cg: "checkout_step",
        bg: "checkout_option",
        li: "coupon",
        mi: "promotions",
        Ja: "user_id",
        wi: "retoken",
        Ia: "cookie_prefix",
        ag: "disable_merchant_reported_purchases",
        ji: "dc_natural_search",
        ii: "dc_custom_params",
        jg: "method",
        xi: "search_term",
        hi: "content_type",
        oi: "optimize_id",
        ni: "experiments",
        qb: "google_signals"
    };
    S.Fd = "google_tld";
    S.Kd = "update";
    S.Me = "firebase_id";
    S.Lc = "ga_restrict_domain";
    S.Cd = "event_settings";
    S.He = "dynamic_event_settings";
    S.ic = "user_data_settings";
    S.sg = "screen_name";
    S.Se = "screen_resolution";
    S.Bb = "_x_19";
    S.ob = "enhanced_client_id";
    S.Ed = "_x_20";
    S.Oe = "internal_traffic_results";
    S.Jd = "traffic_type";
    S.Gd = "referral_exclusion_definition";
    S.Mc = "ignore_referrer";
    S.gi = "content_group";
    S.xa = "allow_interest_groups";
    var $f = {};
    S.xg = Object.freeze(($f[S.aa] = 1, $f[S.zd] = 1, $f[S.Bc] = 1, $f[S.ba] = 1, $f[S.na] = 1, $f[S.za] = 1, $f[S.Ra] = 1, $f[S.$b] = 1, $f[S.nb] = 1, $f[S.Ia] = 1, $f[S.ac] = 1, $f[S.Ic] = 1, $f[S.X] = 1, $f[S.He] = 1, $f[S.bc] = 1, $f[S.Cd] = 1, $f[S.Dd] = 1, $f[S.Kc] = 1, $f[S.Lc] = 1, $f[S.qb] = 1, $f[S.Fd] = 1, $f[S.Ne] = 1, $f[S.Oe] = 1, $f[S.oa] = 1, $f[S.Gd] = 1, $f[S.Pc] = 1, $f[S.Aa] = 1, $f[S.hc] = 1, $f[S.Qc] = 1, $f[S.Hd] = 1, $f[S.Id] = 1, $f[S.Ba] = 1, $f[S.Kd] = 1, $f[S.ic] = 1, $f[S.Ka] = 1, $f[S.Tc] = 1, $f));
    S.wg = Object.freeze([S.fb, S.Sa, S.Oc, S.rb, S.sg, S.Ja, S.Me, S.gi]);
    var ag = {};
    S.Ai = Object.freeze((ag[S.Th] = 1, ag[S.Uh] = 1, ag[S.Vh] = 1, ag[S.Wh] = 1, ag[S.Xh] = 1, ag[S.$h] = 1, ag[S.ai] = 1, ag[S.bi] = 1, ag[S.ci] = 1, ag[S.ye] = 1, ag));
    var bg = {};
    S.yg = Object.freeze((bg[S.Zf] = 1, bg[S.Yh] = 1, bg[S.Wb] = 1, bg[S.Xb] = 1, bg[S.Zh] = 1, bg[S.yb] = 1, bg[S.we] = 1, bg[S.zb] = 1, bg[S.xe] = 1, bg[S.Yb] = 1, bg[S.va] = 1, bg[S.Zb] = 1, bg[S.wa] = 1, bg[S.$f] = 1, bg));
    S.Ue = Object.freeze([S.aa, S.Bc, S.ac]);
    S.Ji = Object.freeze([].concat(S.Ue));
    S.Ve = Object.freeze([S.za, S.Dd, S.Qc, S.Hd, S.Bd]);
    S.Ki = Object.freeze([].concat(S.Ve));
    var cg = {};
    S.ue = (cg[S.D] = "1", cg[S.R] = "2", cg);
    var eg = { Yg: "PH", xh: "PH-40" };
    var fg = {},
        gg = function(a, b) { fg[a] = fg[a] || [];
            fg[a][b] = !0 },
        hg = function(a) { for (var b = [], c = fg[a] || [], d = 0; d < c.length; d++) c[d] && (b[Math.floor(d / 6)] ^= 1 << d % 6); for (var e = 0; e < b.length; e++) b[e] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(b[e] || 0); return b.join("") },
        ig = function() { for (var a = [], b = fg.GA4_EVENT || [], c = 0; c < b.length; c++) b[c] && a.push(c); return 0 < a.length ? a : void 0 };
    var jg = function(a) { gg("GTM", a) };
    var kg = new function(a, b) { this.g = a;
        this.defaultValue = void 0 === b ? !1 : b }(1933);
    var mg = function() { var a = lg,
            b = "xf"; if (a.xf && a.hasOwnProperty(b)) return a.xf; var c = new a;
        a.xf = c;
        a.hasOwnProperty(b); return c };
    var lg = function() { var a = {};
        this.g = function() { var b = kg.g,
                c = kg.defaultValue; return null != a[b] ? a[b] : c };
        this.o = function() { a[kg.g] = !0 } };
    var ng = [];

    function og() { var a = Jb("google_tag_data", {});
        a.ics || (a.ics = { entries: {}, set: pg, update: qg, addListener: rg, notifyListeners: sg, active: !1, usedDefault: !1, usedUpdate: !1, accessedDefault: !1, accessedAny: !1, wasSetLate: !1 }); return a.ics }

    function pg(a, b, c, d, e, f) {
        var g = og();
        !g.usedDefault && g.usedUpdate && (g.wasSetLate = !0);
        g.usedDefault || !g.accessedDefault && !g.accessedAny || (g.wasSetLate = !0);
        g.active = !0;
        g.usedDefault = !0;
        if (void 0 != b) {
            var h = g.entries,
                k = h[a] || {},
                n = k.region,
                p = c && Da(c) ? c.toUpperCase() : void 0;
            d = d.toUpperCase();
            e = e.toUpperCase();
            if ("" === d || p === e || (p === d ? n !== e : !p && !n)) {
                var q = !!(f && 0 < f && void 0 === k.update),
                    r = { region: p, initial: "granted" === b, update: k.update, quiet: q };
                if ("" !== d || !1 !== k.initial) h[a] = r;
                q && m.setTimeout(function() {
                    h[a] ===
                        r && r.quiet && (r.quiet = !1, tg(a), sg(), gg("TAGGING", 2))
                }, f)
            }
        }
    }

    function qg(a, b) { var c = og();
        c.usedDefault || c.usedUpdate || !c.accessedAny || (c.wasSetLate = !0);
        c.active = !0;
        c.usedUpdate = !0; if (void 0 != b) { var d = ug(a),
                e = c.entries,
                f = e[a] = e[a] || {};
            f.update = "granted" === b; var g = ug(a);
            f.quiet ? (f.quiet = !1, tg(a)) : g !== d && tg(a) } }

    function rg(a, b) { ng.push({ nf: a, lj: b }) }

    function tg(a) { for (var b = 0; b < ng.length; ++b) { var c = ng[b];
            Ha(c.nf) && -1 !== c.nf.indexOf(a) && (c.th = !0) } }

    function sg(a, b) { for (var c = 0; c < ng.length; ++c) { var d = ng[c]; if (d.th) { d.th = !1; try { d.lj({ consentEventId: a, consentPriorityId: b }) } catch (e) {} } } }
    var ug = function(a) { var b = og();
            b.accessedAny = !0; var c = b.entries[a] || {}; return void 0 !== c.update ? c.update : c.initial },
        vg = function(a) { var b = og();
            b.accessedDefault = !0; return (b.entries[a] || {}).initial },
        wg = function(a) { var b = og();
            b.accessedAny = !0; return !(b.entries[a] || {}).quiet },
        xg = function() { if (!mg().g()) return !1; var a = og();
            a.accessedAny = !0; return a.active },
        yg = function() { var a = og();
            a.accessedDefault = !0; return a.usedDefault },
        zg = function(a, b) { og().addListener(a, b) },
        Ag = function(a, b) {
            og().notifyListeners(a,
                b)
        },
        Bg = function(a, b) {
            function c() { for (var e = 0; e < b.length; e++)
                    if (!wg(b[e])) return !0;
                return !1 } if (c()) { var d = !1;
                zg(b, function(e) { d || c() || (d = !0, a(e)) }) } else a({}) },
        Cg = function(a, b) {
            function c() { for (var f = [], g = 0; g < d.length; g++) { var h = d[g];!1 === ug(h) || e[h] || (f.push(h), e[h] = !0) } return f } var d = Da(b) ? [b] : b,
                e = {};
            c().length !== d.length && zg(d, function(f) { var g = c();
                0 < g.length && (f.nf = g, a(f)) }) };

    function Dg() {}

    function Eg() {};

    function Fg(a) { for (var b = [], c = 0; c < Gg.length; c++) { var d = a(Gg[c]);
            b[c] = !0 === d ? "1" : !1 === d ? "0" : "-" } return b.join("") }
    var Gg = [S.D, S.R],
        Hg = function(a) { var b = a[S.te];
            b && jg(40); var c = a[S.Tf];
            c && jg(41); for (var d = Ha(b) ? b : [b], e = { xc: 0 }; e.xc < d.length; e = { xc: e.xc }, ++e.xc) Ma(a, function(f) { return function(g, h) { if (g !== S.te && g !== S.Tf) { var k = d[f.xc],
                            n = eg.Yg,
                            p = eg.xh;
                        og().set(g, h, k, n, p, c) } } }(e)) },
        Ig = 0,
        Jg = function(a, b, c) { Ma(a, function(f, g) { og().update(f, g) });
            Ag(b, c && c.priorityId); var d = Ua(),
                e = d - Ig;
            Ig && 0 <= e && 1E3 > e && jg(66);
            Ig = d },
        Kg = function(a) { var b = ug(a); return void 0 != b ? b : !0 },
        Lg = function() { return "G1" + Fg(ug) },
        Mg = function(a, b) {
            zg(a,
                b)
        },
        Ng = function(a, b) { Cg(a, b) },
        Og = function(a, b) { Bg(a, b) };
    var Qg = function(a) { return Pg ? H.querySelectorAll(a) : null },
        Rg = function(a, b) {
            if (!Pg) return null;
            if (Element.prototype.closest) try { return a.closest(b) } catch (e) { return null }
            var c = Element.prototype.matches || Element.prototype.webkitMatchesSelector || Element.prototype.mozMatchesSelector || Element.prototype.msMatchesSelector || Element.prototype.oMatchesSelector,
                d = a;
            if (!H.documentElement.contains(d)) return null;
            do { try { if (c.call(d, b)) return d } catch (e) { break }
                d = d.parentElement || d.parentNode } while (null !== d && 1 === d.nodeType);
            return null
        },
        Sg = !1;
    if (H.querySelectorAll) try { var Tg = H.querySelectorAll(":root");
        Tg && 1 == Tg.length && Tg[0] == H.documentElement && (Sg = !0) } catch (a) {}
    var Pg = Sg;
    var Ug = function(a) { return null == a ? "" : Da(a) ? Ra(String(a)) : "e0" },
        Wg = function(a) { return a.replace(Vg, "") },
        Yg = function(a) { return Xg(a.replace(/\s/g, "")) },
        Xg = function(a) { return Ra(a.replace(Zg, "").toLowerCase()) },
        ah = function(a) { a = a.replace(/[\s-()/.]/g, ""); "+" !== a.charAt(0) && (a = "+" + a); return $g.test(a) ? a : "e0" },
        ch = function(a) { var b = a.toLowerCase().split("@"); if (2 == b.length) { var c = b[0]; /^(gmail|googlemail)\./.test(b[1]) && (c = c.replace(/\./g, ""));
                c = c + "@" + b[1]; if (bh.test(c)) return c } return "e0" },
        fh = function(a,
            b) { window.Promise || b([]);
            Promise.all(a.map(function(c) { return c.value && -1 !== dh.indexOf(c.name) ? eh(c.value).then(function(d) { c.value = d }) : Promise.resolve() })).then(function() { b(a) }).catch(function() { b([]) }) },
        eh = function(a) {
            if ("" === a || "e0" === a) return Promise.resolve(a);
            if (m.crypto && m.crypto.subtle) try {
                var b = gh(a);
                return m.crypto.subtle.digest("SHA-256", b).then(function(c) {
                    var d = Array.from(new Uint8Array(c)).map(function(e) { return String.fromCharCode(e) }).join("");
                    return m.btoa(d).replace(/\+/g, "-").replace(/\//g,
                        "_").replace(/=+$/, "")
                }).catch(function() { return "e2" })
            } catch (c) { return Promise.resolve("e2") } else return Promise.resolve("e1")
        },
        gh = function(a) { var b; if (m.TextEncoder) b = (new m.TextEncoder("utf-8")).encode(a);
            else { for (var c = [], d = 0; d < a.length; d++) { var e = a.charCodeAt(d);
                    128 > e ? c.push(e) : 2048 > e ? c.push(192 | e >> 6, 128 | e & 63) : 55296 > e || 57344 <= e ? c.push(224 | e >> 12, 128 | e >> 6 & 63, 128 | e & 63) : (e = 65536 + ((e & 1023) << 10 | a.charCodeAt(++d) & 1023), c.push(240 | e >> 18, 128 | e >> 12 & 63, 128 | e >> 6 & 63, 128 | e & 63)) }
                b = new Uint8Array(c) } return b },
        Zg = /[0-9`~!@#$%^&*()_\-+=:;<>,.?|/\\[\]]/g,
        bh = /^\S+@\S+\.\S+$/,
        $g = /^\+\d{11,15}$/,
        Vg = /[.~]/g,
        ph = {},
        qh = (ph.email = "em", ph.phone_number = "pn", ph.first_name = "fn", ph.last_name = "ln", ph.street = "sa", ph.city = "ct", ph.region = "rg", ph.country = "co", ph.postal_code = "pc", ph.error_code = "ec", ph),
        rh = function(a, b) {
            function c(n, p, q) { var r = n[p];
                Ha(r) || (r = [r]); for (var u = 0; u < r.length; ++u) { var t = Ug(r[u]); "" !== t && f.push({ name: p, value: q(t), index: void 0 }) } }

            function d(n, p, q, r) { var u = Ug(n[p]); "" !== u && f.push({ name: p, value: q(u), index: r }) }

            function e(n) { return function(p) { jg(64); return n(p) } }
            var f = [];
            if ("https:" === m.location.protocol) {
                c(a, "email", ch);
                c(a, "phone_number", ah);
                c(a, "first_name", e(Yg));
                c(a, "last_name", e(Yg));
                var g = a.home_address || {};
                c(g, "street", e(Xg));
                c(g, "city", e(Xg));
                c(g, "postal_code", e(Wg));
                c(g, "region", e(Xg));
                c(g, "country", e(Wg));
                var h = a.address || {};
                Ha(h) || (h = [h]);
                for (var k = 0; k < h.length; k++) d(h[k], "first_name", Yg, k), d(h[k], "last_name", Yg, k), d(h[k], "street", Xg, k), d(h[k], "city", Xg, k), d(h[k], "postal_code", Wg, k), d(h[k],
                    "region", Xg, k), d(h[k], "country", Wg, k);
                fh(f, b)
            } else f.push({ name: "error_code", value: "e3", index: void 0 }), b(f)
        },
        sh = function(a, b) { rh(a, function(c) { for (var d = ["tv.1"], e = 0, f = 0; f < c.length; ++f) { var g = c[f].name,
                        h = c[f].value,
                        k = c[f].index,
                        n = qh[g];
                    n && h && (-1 === dh.indexOf(g) || /^e\d+$/.test(h) || /^[0-9A-Za-z_-]{43}$/.test(h)) && (void 0 !== k && (n += k), d.push(n + "." + h), e++) }
                1 === c.length && "error_code" === c[0].name && (e = 0);
                b(encodeURIComponent(d.join("~")), e) }) },
        th = function(a) {
            if (m.Promise) try {
                return new Promise(function(b) {
                    sh(a,
                        function(c, d) { b({ ie: c, Ik: d }) })
                })
            } catch (b) {}
        },
        dh = Object.freeze(["email", "phone_number", "first_name", "last_name", "street"]);
    var J = {},
        T = m.google_tag_manager = m.google_tag_manager || {},
        uh = Math.random();
    J.J = "G-CD35DZJ728";
    J.xd = "";
    J.Pd = "370";
    J.Z = "dataLayer";
    J.Sh = "ChAIgIOskQYQ7fyJzamLoNpzEiUAYYdxRs7er9PrYwJIOW6tFPTS1NNygWlMpCQEt3ZJq2+Wa4eRGgKt6Q\x3d\x3d";
    var vh = { __cl: !0, __ecl: !0, __ehl: !0, __evl: !0, __fal: !0, __fil: !0, __fsl: !0, __hl: !0, __jel: !0, __lcl: !0, __sdl: !0, __tl: !0, __ytl: !0 },
        wh = { __paused: !0, __tg: !0 },
        xh;
    for (xh in vh) vh.hasOwnProperty(xh) && (wh[xh] = !0);
    J.yd = "www.googletagmanager.com";
    var yh, zh = J.yd + "/gtm.js";
    zh = J.yd + "/gtag/js";
    yh = zh;
    var Ah = Pa(""),
        Bh = null,
        Ch = null,
        Dh = "https://www.googletagmanager.com/a?id=" + J.J + "&cv=1",
        Eh = {},
        Fh = {},
        Gh = function() { var a = T.sequence || 1;
            T.sequence = a + 1; return a };
    J.Rh = "";
    var Hh = "";
    J.Vc = Hh;
    var Ih = new Ka,
        Jh = {},
        Kh = {},
        Nh = { name: J.Z, set: function(a, b) { pc(ab(a, b), Jh);
                Lh() }, get: function(a) { return Mh(a, 2) }, reset: function() { Ih = new Ka;
                Jh = {};
                Lh() } },
        Mh = function(a, b) { return 2 != b ? Ih.get(a) : Oh(a) },
        Oh = function(a, b) { var c = a.split(".");
            b = b || []; for (var d = Jh, e = 0; e < c.length; e++) { if (null === d) return !1; if (void 0 === d) break;
                d = d[c[e]]; if (-1 !== b.indexOf(d)) return } return d },
        Ph = function(a, b) { Kh.hasOwnProperty(a) || (Ih.set(a, b), pc(ab(a, b), Jh), Lh()) },
        Qh = function() {
            for (var a = ["gtm.allowlist", "gtm.blocklist", "gtm.whitelist",
                    "gtm.blacklist", "tagTypeBlacklist"
                ], b = 0; b < a.length; b++) { var c = a[b],
                    d = Mh(c, 1); if (Ha(d) || oc(d)) d = pc(d);
                Kh[c] = d }
        },
        Lh = function(a) { Ma(Kh, function(b, c) { Ih.set(b, c);
                pc(ab(b, void 0), Jh);
                pc(ab(b, c), Jh);
                a && delete Kh[b] }) },
        Rh = function(a, b) { var c, d = 1 !== (void 0 === b ? 2 : b) ? Oh(a) : Ih.get(a); "array" === mc(d) || "object" === mc(d) ? c = pc(d) : c = d; return c };
    var Sh, Th = !1;

    function Uh() { Th = !0;
        Sh = Sh || {} }
    var Vh = function(a) { Th || Uh(); return Sh[a] };
    var Wh = function(a) {
        if (H.hidden) return !0;
        var b = a.getBoundingClientRect();
        if (b.top == b.bottom || b.left == b.right || !m.getComputedStyle) return !0;
        var c = m.getComputedStyle(a, null);
        if ("hidden" === c.visibility) return !0;
        for (var d = a, e = c; d;) {
            if ("none" === e.display) return !0;
            var f = e.opacity,
                g = e.filter;
            if (g) { var h = g.indexOf("opacity(");
                0 <= h && (g = g.substring(h + 8, g.indexOf(")", h)), "%" == g.charAt(g.length - 1) && (g = g.substring(0, g.length - 1)), f = Math.min(g, f)) }
            if (void 0 !== f && 0 >= f) return !0;
            (d = d.parentElement) && (e = m.getComputedStyle(d,
                null))
        }
        return !1
    };
    var Xh = function() { var a = H.body,
                b = H.documentElement || a && a.parentElement,
                c, d; if (H.compatMode && "BackCompat" !== H.compatMode) c = b ? b.clientHeight : 0, d = b ? b.clientWidth : 0;
            else { var e = function(f, g) { return f && g ? Math.min(f, g) : Math.max(f, g) };
                jg(7);
                c = e(b ? b.clientHeight : 0, a ? a.clientHeight : 0);
                d = e(b ? b.clientWidth : 0, a ? a.clientWidth : 0) } return { width: d, height: c } },
        Yh = function(a) {
            var b = Xh(),
                c = b.height,
                d = b.width,
                e = a.getBoundingClientRect(),
                f = e.bottom - e.top,
                g = e.right - e.left;
            return f && g ? (1 - Math.min((Math.max(0 - e.left, 0) + Math.max(e.right -
                d, 0)) / g, 1)) * (1 - Math.min((Math.max(0 - e.top, 0) + Math.max(e.bottom - c, 0)) / f, 1)) : 0
        };
    var ei = /:[0-9]+$/,
        fi = function(a, b, c, d) { for (var e = [], f = a.split("&"), g = 0; g < f.length; g++) { var h = f[g].split("="); if (decodeURIComponent(h[0]).replace(/\+/g, " ") === b) { var k = h.slice(1).join("="); if (!c) return d ? k : decodeURIComponent(k).replace(/\+/g, " ");
                    e.push(d ? k : decodeURIComponent(k).replace(/\+/g, " ")) } } return c ? e : void 0 },
        ii = function(a, b, c, d, e) {
            b && (b = String(b).toLowerCase());
            if ("protocol" === b || "port" === b) a.protocol = gi(a.protocol) || gi(m.location.protocol);
            "port" === b ? a.port = String(Number(a.hostname ? a.port :
                m.location.port) || ("http" === a.protocol ? 80 : "https" === a.protocol ? 443 : "")) : "host" === b && (a.hostname = (a.hostname || m.location.hostname).replace(ei, "").toLowerCase());
            return hi(a, b, c, d, e)
        },
        hi = function(a, b, c, d, e) {
            var f, g = gi(a.protocol);
            b && (b = String(b).toLowerCase());
            switch (b) {
                case "url_no_fragment":
                    f = ji(a);
                    break;
                case "protocol":
                    f = g;
                    break;
                case "host":
                    f = a.hostname.replace(ei, "").toLowerCase();
                    if (c) { var h = /^www\d*\./.exec(f);
                        h && h[0] && (f = f.substr(h[0].length)) }
                    break;
                case "port":
                    f = String(Number(a.port) || ("http" ===
                        g ? 80 : "https" === g ? 443 : ""));
                    break;
                case "path":
                    a.pathname || a.hostname || gg("TAGGING", 1);
                    f = "/" === a.pathname.substr(0, 1) ? a.pathname : "/" + a.pathname;
                    var k = f.split("/");
                    0 <= (d || []).indexOf(k[k.length - 1]) && (k[k.length - 1] = "");
                    f = k.join("/");
                    break;
                case "query":
                    f = a.search.replace("?", "");
                    e && (f = fi(f, e, !1, void 0));
                    break;
                case "extension":
                    var n = a.pathname.split(".");
                    f = 1 < n.length ? n[n.length - 1] : "";
                    f = f.split("/")[0];
                    break;
                case "fragment":
                    f = a.hash.replace("#", "");
                    break;
                default:
                    f = a && a.href
            }
            return f
        },
        gi = function(a) {
            return a ?
                a.replace(":", "").toLowerCase() : ""
        },
        ji = function(a) { var b = ""; if (a && a.href) { var c = a.href.indexOf("#");
                b = 0 > c ? a.href : a.href.substr(0, c) } return b },
        ki = function(a) { var b = H.createElement("a");
            a && (b.href = a); var c = b.pathname; "/" !== c[0] && (a || gg("TAGGING", 1), c = "/" + c); var d = b.hostname.replace(ei, ""); return { href: b.href, protocol: b.protocol, host: b.host, hostname: d, pathname: c, search: b.search, hash: b.hash, port: b.port } },
        li = function(a) {
            function b(n) { var p = n.split("=")[0]; return 0 > d.indexOf(p) ? n : p + "=0" }

            function c(n) {
                return n.split("&").map(b).filter(function(p) {
                    return void 0 !==
                        p
                }).join("&")
            }
            var d = "gclid dclid gbraid wbraid gclaw gcldc gclha gclgf gclgb _gl".split(" "),
                e = ki(a),
                f = a.split(/[?#]/)[0],
                g = e.search,
                h = e.hash;
            "?" === g[0] && (g = g.substring(1));
            "#" === h[0] && (h = h.substring(1));
            g = c(g);
            h = c(h);
            "" !== g && (g = "?" + g);
            "" !== h && (h = "#" + h);
            var k = "" + f + g + h;
            "/" === k[k.length - 1] && (k = k.substring(0, k.length - 1));
            return k
        };
    var mi = {};
    var pi = function(a) { if (0 == a.length) return null; var b;
            b = ni(a, function(c) { return !oi.test(c.Za) });
            b = ni(b, function(c) { return "INPUT" === c.element.tagName.toUpperCase() });
            b = ni(b, function(c) { return !Wh(c.element) }); return b[0] },
        ni = function(a, b) { if (1 >= a.length) return a; var c = a.filter(b); return 0 == c.length ? a : c },
        qi = function(a) {
            var b;
            if (a === H.body) b = "body";
            else {
                var c;
                if (a.id) c = "#" + a.id;
                else {
                    var d;
                    if (a.parentElement) {
                        var e;
                        a: {
                            var f = a.parentElement;
                            if (f) {
                                for (var g = 0; g < f.childElementCount; g++)
                                    if (f.children[g] === a) {
                                        e =
                                            g + 1;
                                        break a
                                    }
                                e = -1
                            } else e = 1
                        }
                        d = qi(a.parentElement) + ">:nth-child(" + e + ")"
                    } else d = "";
                    c = d
                }
                b = c
            }
            return b
        },
        ri = !0,
        si = !1;
    mi.Oh = "true";
    var ti = new RegExp(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i),
        ui = new RegExp(/@(gmail|googlemail)\./i),
        oi = new RegExp(/support|noreply/i),
        vi = "SCRIPT STYLE IMG SVG PATH BR NOSCRIPT".split(" "),
        wi = ["BR"],
        xi = {},
        yi = function(a) {
            a = a || { fe: !0, he: !0 };
            a.vb = a.vb || { email: !0, phone: !0, Rg: !0 };
            var b, c = a,
                d = !!c.fe + "." + !!c.he;
            c && c.$c && c.$c.length && (d += "." + c.$c.join("."));
            c && c.vb && (d += "." + c.vb.email + "." + c.vb.phone + "." + c.vb.Rg);
            b = d;
            var e = xi[b];
            if (e && 200 > Ua() - e.timestamp) return e.result;
            var f;
            var g = [],
                h = H.body;
            if (h) {
                for (var k = h.querySelectorAll("*"), n = 0; n < k.length && 1E4 > n; n++) {
                    var p = k[n];
                    if (!(0 <= vi.indexOf(p.tagName.toUpperCase())) &&
                        p.children instanceof HTMLCollection) { for (var q = !1, r = 0; r < p.childElementCount && 1E4 > r; r++)
                            if (!(0 <= wi.indexOf(p.children[r].tagName.toUpperCase()))) { q = !0; break }
                        q || g.push(p) }
                }
                f = { elements: g, status: 1E4 < k.length ? "2" : "1" }
            } else f = { elements: g, status: "4" };
            var u = f,
                t = u.status,
                v;
            if (a.vb && a.vb.email) {
                for (var x = u.elements, y = [], w = 0; w < x.length; w++) {
                    var A = x[w],
                        B = A.textContent;
                    "INPUT" === A.tagName.toUpperCase() && A.value && (B = A.value);
                    if (B) {
                        var C = B.match(ti);
                        if (C) {
                            var D = C[0],
                                E;
                            if (m.location) {
                                var F = hi(m.location, "host", !0);
                                E = 0 <= D.toLowerCase().indexOf(F)
                            } else E = !1;
                            E || y.push({ element: A, Za: D })
                        }
                    }
                }
                var P;
                var K = a && a.$c;
                if (K && 0 !== K.length) { for (var Q = [], R = 0; R < y.length; R++) { for (var O = !0, L = 0; L < K.length; L++) { var ba = K[L]; if (ba && Rg(y[R].element, ba)) { O = !1; break } }
                        O && Q.push(y[R]) }
                    P = Q } else P = y;
                v = pi(P);
                10 < y.length && (t = "3")
            }
            var X = [];
            if (v) { var W = v.element,
                    va = { Za: v.Za, tagName: W.tagName, type: 1 };
                a.fe && (va.querySelector = qi(W));
                a.he && (va.isVisible = !Wh(W));
                X.push(va) }
            var Ea = { elements: X, status: t };
            xi[b] = { timestamp: Ua(), result: Ea };
            return Ea
        };

    var zi = function(a, b, c) {
            if (c) {
                var d = c.selector_type,
                    e = String(c.value),
                    f;
                if ("js_variable" === d) { e = e.replace(/\["?'?/g, ".").replace(/"?'?\]/g, ""); for (var g = e.split(","), h = 0; h < g.length; h++) { var k = g[h].trim(); if (k) { if (0 === k.indexOf("dataLayer.")) f = Mh(k.substring(10));
                            else { var n = k.split(".");
                                f = m[n.shift()]; for (var p = 0; p < n.length; p++) f = f && f[n[p]] } if (void 0 !== f) break } } } else if ("css_selector" === d && Pg) {
                    var q = Qg(e);
                    if (q && 0 < q.length) {
                        f = [];
                        for (var r = 0; r < q.length && r < ("email" === b || "phone_number" === b ? 5 : 1); r++) f.push(Vb(q[r]) ||
                            Ra(q[r].value));
                        f = 1 === f.length ? f[0] : f
                    }
                }
                f && (a[b] = f)
            }
        },
        Ai = function(a) { if (a) { var b = {};
                zi(b, "email", a.email);
                zi(b, "phone_number", a.phone);
                b.address = []; for (var c = a.name_and_address || [], d = 0; d < c.length; d++) { var e = {};
                    zi(e, "first_name", c[d].first_name);
                    zi(e, "last_name", c[d].last_name);
                    zi(e, "street", c[d].street);
                    zi(e, "city", c[d].city);
                    zi(e, "region", c[d].region);
                    zi(e, "country", c[d].country);
                    zi(e, "postal_code", c[d].postal_code);
                    b.address.push(e) } return b } },
        Bi = function(a) {
            if (a) switch (a.mode) {
                case "selectors":
                    return Ai(a.selectors);
                case "auto_detect":
                    var b;
                    var c = a.auto_detect;
                    if (c) { var d = yi({ fe: !1, he: !1, $c: c.exclude_element_selectors, vb: { email: !!c.email, phone: !!c.phone, Rg: !!c.address } }).elements,
                            e = {}; if (0 < d.length)
                            for (var f = 0; f < d.length; f++) { var g = d[f]; if (1 === g.type) { e.email = g.Za; break } }
                        b = e } else b = void 0;
                    return b
            }
        },
        Ci = function(a) { switch (a.enhanced_conversions_mode) {
                case "manual":
                    var b = a.enhanced_conversions_manual_var; return void 0 !== b ? b : m.enhanced_conversion_data;
                case "automatic":
                    return Ai(a[S.eg]) } };
    var Di = {},
        Ei = function(a, b) { if (m._gtmexpgrp && m._gtmexpgrp.hasOwnProperty(a)) return m._gtmexpgrp[a];
            void 0 === Di[a] && (Di[a] = Math.floor(Math.random() * b)); return Di[a] };
    var Fi = function(a) { var b = 1,
            c, d, e; if (a)
            for (b = 0, d = a.length - 1; 0 <= d; d--) e = a.charCodeAt(d), b = (b << 6 & 268435455) + e + (e << 14), c = b & 266338304, b = 0 !== c ? b ^ c >> 21 : b; return b };
    var Gi = function(a, b, c) { for (var d = [], e = b.split(";"), f = 0; f < e.length; f++) { var g = e[f].split("="),
                h = g[0].replace(/^\s*|\s*$/g, ""); if (h && h == a) { var k = g.slice(1).join("=").replace(/^\s*|\s*$/g, "");
                k && c && (k = decodeURIComponent(k));
                d.push(k) } } return d };
    var Hi = function(a, b) { var c = function() {};
            c.prototype = a.prototype; var d = new c;
            a.apply(d, Array.prototype.slice.call(arguments, 1)); return d },
        Ii = function(a) { var b = a; return function() { if (b) { var c = b;
                    b = null;
                    c() } } };

    function Ji(a) { return "null" !== a.origin };
    var Mi = function(a, b, c, d) { return Ki(d) ? Gi(a, String(b || Li()), c) : [] },
        Pi = function(a, b, c, d, e) { if (Ki(e)) { var f = Ni(a, d, e); if (1 === f.length) return f[0].id; if (0 !== f.length) { f = Oi(f, function(g) { return g.Yd }, b); if (1 === f.length) return f[0].id;
                    f = Oi(f, function(g) { return g.jd }, c); return f[0] ? f[0].id : void 0 } } };

    function Qi(a, b, c, d) { var e = Li(),
            f = window;
        Ji(f) && (f.document.cookie = a); var g = Li(); return e != g || void 0 != c && 0 <= Mi(b, g, !1, d).indexOf(c) }
    var Ui = function(a, b, c, d) {
            function e(x, y, w) { if (null == w) return delete h[y], x;
                h[y] = w; return x + "; " + y + "=" + w }

            function f(x, y) { if (null == y) return delete h[y], x;
                h[y] = !0; return x + "; " + y }
            if (!Ki(c.jb)) return 2;
            var g;
            void 0 == b ? g = a + "=deleted; expires=" + (new Date(0)).toUTCString() : (c.encode && (b = encodeURIComponent(b)), b = Ri(b), g = a + "=" + b);
            var h = {};
            g = e(g, "path", c.path);
            var k;
            c.expires instanceof Date ? k = c.expires.toUTCString() : null != c.expires && (k = "" + c.expires);
            g = e(g, "expires", k);
            g = e(g, "max-age", c.Lj);
            g = e(g, "samesite",
                c.ek);
            c.gk && (g = f(g, "secure"));
            var n = c.domain;
            if (n && "auto" === n.toLowerCase()) { for (var p = Si(), q = void 0, r = !1, u = 0; u < p.length; ++u) { var t = "none" !== p[u] ? p[u] : void 0,
                        v = e(g, "domain", t);
                    v = f(v, c.flags); try { d && d(a, h) } catch (x) { q = x; continue }
                    r = !0; if (!Ti(t, c.path) && Qi(v, a, b, c.jb)) return 0 } if (q && !r) throw q; return 1 }
            n && "none" !== n.toLowerCase() && (g = e(g, "domain", n));
            g = f(g, c.flags);
            d && d(a, h);
            return Ti(n, c.path) ? 1 : Qi(g, a, b, c.jb) ? 0 : 1
        },
        Vi = function(a, b, c) {
            null == c.path && (c.path = "/");
            c.domain || (c.domain = "auto");
            return Ui(a,
                b, c)
        };

    function Oi(a, b, c) { for (var d = [], e = [], f, g = 0; g < a.length; g++) { var h = a[g],
                k = b(h);
            k === c ? d.push(h) : void 0 === f || k < f ? (e = [h], f = k) : k === f && e.push(h) } return 0 < d.length ? d : e }

    function Ni(a, b, c) { for (var d = [], e = Mi(a, void 0, void 0, c), f = 0; f < e.length; f++) { var g = e[f].split("."),
                h = g.shift(); if (!b || -1 !== b.indexOf(h)) { var k = g.shift();
                k && (k = k.split("-"), d.push({ id: g.join("."), Yd: 1 * k[0] || 1, jd: 1 * k[1] || 1 })) } } return d }
    var Ri = function(a) { a && 1200 < a.length && (a = a.substring(0, 1200)); return a },
        Wi = /^(www\.)?google(\.com?)?(\.[a-z]{2})?$/,
        Xi = /(^|\.)doubleclick\.net$/i,
        Ti = function(a, b) { return Xi.test(window.document.location.hostname) || "/" === b && Wi.test(a) },
        Li = function() { return Ji(window) ? window.document.cookie : "" },
        Si = function() {
            var a = [],
                b = window.document.location.hostname.split(".");
            if (4 === b.length) { var c = b[b.length - 1]; if (parseInt(c, 10).toString() === c) return ["none"] }
            for (var d = b.length - 2; 0 <= d; d--) a.push(b.slice(d).join("."));
            var e = window.document.location.hostname;
            Xi.test(e) || Wi.test(e) || a.push("none");
            return a
        },
        Ki = function(a) { if (!mg().g() || !a || !xg()) return !0; if (!wg(a)) return !1; var b = ug(a); return null == b ? !0 : !!b };
    var Yi = function(a) { var b = Math.round(2147483647 * Math.random()); return a ? String(b ^ Fi(a) & 2147483647) : String(b) },
        Zi = function(a) { return [Yi(a), Math.round(Ua() / 1E3)].join(".") },
        bj = function(a, b, c, d, e) { var f = $i(b); return Pi(a, f, aj(c), d, e) },
        cj = function(a, b, c, d) { var e = "" + $i(c),
                f = aj(d);
            1 < f && (e += "-" + f); return [b, e, a].join(".") },
        $i = function(a) { if (!a) return 1;
            a = 0 === a.indexOf(".") ? a.substr(1) : a; return a.split(".").length },
        aj = function(a) {
            if (!a || "/" === a) return 1;
            "/" !== a[0] && (a = "/" + a);
            "/" !== a[a.length - 1] && (a += "/");
            return a.split("/").length -
                1
        };

    function dj(a, b, c) { var d, e = Number(null != a.Pb ? a.Pb : void 0);
        0 !== e && (d = new Date((b || Ua()) + 1E3 * (e || 7776E3))); return { path: a.path, domain: a.domain, flags: a.flags, encode: !!c, expires: d } };
    var ej = ["1"],
        fj = {},
        gj = {},
        kj = function(a, b) { b = void 0 === b ? !0 : b; var c = hj(a.prefix); if (!fj[c] && !ij(c, a.path, a.domain) && b) { var d = hj(a.prefix),
                    e = Zi(); if (0 === jj(d, e, a)) { var f = Jb("google_tag_data", {});
                    f._gcl_au ? gg("GTM", 57) : f._gcl_au = e }
                ij(c, a.path, a.domain) } };

    function jj(a, b, c, d) { var e = cj(b, "1", c.domain, c.path),
            f = dj(c, d);
        f.jb = "ad_storage"; return Vi(a, e, f) }

    function ij(a, b, c) { var d = bj(a, b, c, ej, "ad_storage"); if (!d) return !1; var e = d.split(".");
        5 === e.length ? (fj[a] = e.slice(0, 2).join("."), gj[a] = { id: e.slice(2, 4).join("."), nh: Number(e[4]) || 0 }) : 3 === e.length ? gj[a] = { id: e.slice(0, 2).join("."), nh: Number(e[2]) || 0 } : fj[a] = d; return !0 }

    function hj(a) { return (a || "_gcl") + "_au" };
    var lj = function(a) { for (var b = [], c = H.cookie.split(";"), d = new RegExp("^\\s*" + (a || "_gac") + "_(UA-\\d+-\\d+)=\\s*(.+?)\\s*$"), e = 0; e < c.length; e++) { var f = c[e].match(d);
            f && b.push({ Qf: f[1], value: f[2], timestamp: Number(f[2].split(".")[1]) || 0 }) }
        b.sort(function(g, h) { return h.timestamp - g.timestamp }); return b };

    function mj(a, b) { var c = lj(a),
            d = {}; if (!c || !c.length) return d; for (var e = 0; e < c.length; e++) { var f = c[e].value.split("."); if (!("1" !== f[0] || b && 3 > f.length || !b && 3 !== f.length) && Number(f[1])) { d[c[e].Qf] || (d[c[e].Qf] = []); var g = { version: f[0], timestamp: 1E3 * Number(f[1]), Da: f[2] };
                b && 3 < f.length && (g.labels = f.slice(3));
                d[c[e].Qf].push(g) } } return d };

    function nj() { for (var a = oj, b = {}, c = 0; c < a.length; ++c) b[a[c]] = c; return b }

    function pj() { var a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        a += a.toLowerCase() + "0123456789-_"; return a + "." }
    var oj, qj;

    function rj(a) {
        function b(k) { for (; d < a.length;) { var n = a.charAt(d++),
                    p = qj[n]; if (null != p) return p; if (!/^[\s\xa0]*$/.test(n)) throw Error("Unknown base64 encoding at char: " + n); } return k }
        oj = oj || pj();
        qj = qj || nj(); for (var c = "", d = 0;;) { var e = b(-1),
                f = b(0),
                g = b(64),
                h = b(64); if (64 === h && -1 === e) return c;
            c += String.fromCharCode(e << 2 | f >> 4);
            64 != g && (c += String.fromCharCode(f << 4 & 240 | g >> 2), 64 != h && (c += String.fromCharCode(g << 6 & 192 | h))) } };
    var sj;
    var wj = function() { var a = tj,
                b = uj,
                c = vj(),
                d = function(g) { a(g.target || g.srcElement || {}) },
                e = function(g) { b(g.target || g.srcElement || {}) }; if (!c.init) { Sb(H, "mousedown", d);
                Sb(H, "keyup", d);
                Sb(H, "submit", e); var f = HTMLFormElement.prototype.submit;
                HTMLFormElement.prototype.submit = function() { b(this);
                    f.call(this) };
                c.init = !0 } },
        xj = function(a, b, c, d, e) { var f = { callback: a, domains: b, fragment: 2 === c, placement: c, forms: d, sameHost: e };
            vj().decorators.push(f) },
        yj = function(a, b, c) {
            for (var d = vj().decorators, e = {}, f = 0; f < d.length; ++f) {
                var g =
                    d[f],
                    h;
                if (h = !c || g.forms) a: { var k = g.domains,
                        n = a,
                        p = !!g.sameHost; if (k && (p || n !== H.location.hostname))
                        for (var q = 0; q < k.length; q++)
                            if (k[q] instanceof RegExp) { if (k[q].test(n)) { h = !0; break a } } else if (0 <= n.indexOf(k[q]) || p && 0 <= k[q].indexOf(n)) { h = !0; break a }
                    h = !1 }
                if (h) { var r = g.placement;
                    void 0 == r && (r = g.fragment ? 2 : 1);
                    r === b && Xa(e, g.callback()) }
            }
            return e
        };

    function vj() { var a = Jb("google_tag_data", {}),
            b = a.gl;
        b && b.decorators || (b = { decorators: [] }, a.gl = b); return b };
    var zj = /(.*?)\*(.*?)\*(.*)/,
        Aj = /^https?:\/\/([^\/]*?)\.?cdn\.ampproject\.org\/?(.*)/,
        Bj = /^(?:www\.|m\.|amp\.)+/,
        Cj = /([^?#]+)(\?[^#]*)?(#.*)?/;

    function Dj(a) { return new RegExp("(.*?)(^|&)" + a + "=([^&]*)&?(.*)") }
    var Fj = function(a) {
        var b = [],
            c;
        for (c in a)
            if (a.hasOwnProperty(c)) { var d = a[c]; if (void 0 !== d && d === d && null !== d && "[object Object]" !== d.toString()) { b.push(c); var e = b,
                        f = e.push,
                        g, h = String(d);
                    oj = oj || pj();
                    qj = qj || nj(); for (var k = [], n = 0; n < h.length; n += 3) { var p = n + 1 < h.length,
                            q = n + 2 < h.length,
                            r = h.charCodeAt(n),
                            u = p ? h.charCodeAt(n + 1) : 0,
                            t = q ? h.charCodeAt(n + 2) : 0,
                            v = r >> 2,
                            x = (r & 3) << 4 | u >> 4,
                            y = (u & 15) << 2 | t >> 6,
                            w = t & 63;
                        q || (w = 64, p || (y = 64));
                        k.push(oj[v], oj[x], oj[y], oj[w]) }
                    g = k.join("");
                    f.call(e, g) } }
        var A = b.join("*");
        return ["1", Ej(A),
            A
        ].join("*")
    };

    function Ej(a, b) { var c = [m.navigator.userAgent, (new Date).getTimezoneOffset(), Hb.userLanguage || Hb.language, Math.floor(Ua() / 60 / 1E3) - (void 0 === b ? 0 : b), a].join("*"),
            d; if (!(d = sj)) { for (var e = Array(256), f = 0; 256 > f; f++) { for (var g = f, h = 0; 8 > h; h++) g = g & 1 ? g >>> 1 ^ 3988292384 : g >>> 1;
                e[f] = g }
            d = e }
        sj = d; for (var k = 4294967295, n = 0; n < c.length; n++) k = k >>> 8 ^ sj[(k ^ c.charCodeAt(n)) & 255]; return ((k ^ -1) >>> 0).toString(36) }

    function Gj() { return function(a) { var b = ki(m.location.href),
                c = b.search.replace("?", ""),
                d = fi(c, "_gl", !1, !0) || "";
            a.query = Hj(d) || {}; var e = ii(b, "fragment").match(Dj("_gl"));
            a.fragment = Hj(e && e[3] || "") || {} } }

    function Ij(a, b) { var c = Dj(a).exec(b),
            d = b; if (c) { var e = c[2],
                f = c[4];
            d = c[1];
            f && (d = d + e + f) } return d }
    var Jj = function(a, b) { b || (b = "_gl"); var c = Cj.exec(a); if (!c) return ""; var d = c[1],
                e = Ij(b, (c[2] || "").slice(1)),
                f = Ij(b, (c[3] || "").slice(1));
            e.length && (e = "?" + e);
            f.length && (f = "#" + f); return "" + d + e + f },
        Kj = function(a) { var b = Gj(),
                c = vj();
            c.data || (c.data = { query: {}, fragment: {} }, b(c.data)); var d = {},
                e = c.data;
            e && (Xa(d, e.query), a && Xa(d, e.fragment)); return d },
        Hj = function(a) {
            try { var b = Lj(a, 3); if (void 0 !== b) { for (var c = {}, d = b ? b.split("*") : [], e = 0; e + 1 < d.length; e += 2) { var f = d[e],
                            g = rj(d[e + 1]);
                        c[f] = g }
                    gg("TAGGING", 6); return c } } catch (h) {
                gg("TAGGING",
                    8)
            }
        };

    function Lj(a, b) { if (a) { var c;
            a: { for (var d = a, e = 0; 3 > e; ++e) { var f = zj.exec(d); if (f) { c = f; break a }
                    d = decodeURIComponent(d) }
                c = void 0 }
            var g = c; if (g && "1" === g[1]) { var h = g[3],
                    k;
                a: { for (var n = g[2], p = 0; p < b; ++p)
                        if (n === Ej(h, p)) { k = !0; break a }
                    k = !1 }
                if (k) return h;
                gg("TAGGING", 7) } } }

    function Mj(a, b, c, d) {
        function e(p) { p = Ij(a, p); var q = p.charAt(p.length - 1);
            p && "&" !== q && (p += "&"); return p + n }
        d = void 0 === d ? !1 : d; var f = Cj.exec(c); if (!f) return ""; var g = f[1],
            h = f[2] || "",
            k = f[3] || "",
            n = a + "=" + b;
        d ? k = "#" + e(k.substring(1)) : h = "?" + e(h.substring(1)); return "" + g + h + k }

    function Nj(a, b) { var c = "FORM" === (a.tagName || "").toUpperCase(),
            d = yj(b, 1, c),
            e = yj(b, 2, c),
            f = yj(b, 3, c); if (Ya(d)) { var g = Fj(d);
            c ? Oj("_gl", g, a) : Pj("_gl", g, a, !1) } if (!c && Ya(e)) { var h = Fj(e);
            Pj("_gl", h, a, !0) } for (var k in f)
            if (f.hasOwnProperty(k)) a: { var n = k,
                    p = f[k],
                    q = a; if (q.tagName) { if ("a" === q.tagName.toLowerCase()) { Pj(n, p, q, void 0); break a } if ("form" === q.tagName.toLowerCase()) { Oj(n, p, q); break a } } "string" == typeof q && Mj(n, p, q, void 0) } }

    function Pj(a, b, c, d) { if (c.href) { var e = Mj(a, b, c.href, void 0 === d ? !1 : d);
            tb.test(e) && (c.href = e) } }

    function Oj(a, b, c) { if (c && c.action) { var d = (c.method || "").toLowerCase(); if ("get" === d) { for (var e = c.childNodes || [], f = !1, g = 0; g < e.length; g++) { var h = e[g]; if (h.name === a) { h.setAttribute("value", b);
                        f = !0; break } } if (!f) { var k = H.createElement("input");
                    k.setAttribute("type", "hidden");
                    k.setAttribute("name", a);
                    k.setAttribute("value", b);
                    c.appendChild(k) } } else if ("post" === d) { var n = Mj(a, b, c.action);
                tb.test(n) && (c.action = n) } } }

    function tj(a) { try { var b;
            a: { for (var c = a, d = 100; c && 0 < d;) { if (c.href && c.nodeName.match(/^a(?:rea)?$/i)) { b = c; break a }
                    c = c.parentNode;
                    d-- }
                b = null }
            var e = b; if (e) { var f = e.protocol; "http:" !== f && "https:" !== f || Nj(e, e.hostname) } } catch (g) {} }

    function uj(a) { try { if (a.action) { var b = ii(ki(a.action), "host");
                Nj(a, b) } } catch (c) {} }
    var Qj = function(a, b, c, d) { wj();
            xj(a, b, "fragment" === c ? 2 : 1, !!d, !1) },
        Rj = function(a, b) { wj();
            xj(a, [hi(m.location, "host", !0)], b, !0, !0) },
        Sj = function() {
            var a = H.location.hostname,
                b = Aj.exec(H.referrer);
            if (!b) return !1;
            var c = b[2],
                d = b[1],
                e = "";
            if (c) { var f = c.split("/"),
                    g = f[1];
                e = "s" === g ? decodeURIComponent(f[2]) : decodeURIComponent(g) } else if (d) { if (0 === d.indexOf("xn--")) return !1;
                e = d.replace(/-/g, ".").replace(/\.\./g, "-") }
            var h = a.replace(Bj, ""),
                k = e.replace(Bj, ""),
                n;
            if (!(n = h === k)) {
                var p = "." + k;
                n = h.substring(h.length - p.length,
                    h.length) === p
            }
            return n
        },
        Tj = function(a, b) { return !1 === a ? !1 : a || b || Sj() };
    var Uj = {};
    var Vj = /^\w+$/,
        Wj = /^[\w-]+$/,
        Xj = { aw: "_aw", dc: "_dc", gf: "_gf", ha: "_ha", gp: "_gp", gb: "_gb" },
        Yj = function() { if (!mg().g() || !xg()) return !0; var a = ug("ad_storage"); return null == a ? !0 : !!a },
        Zj = function(a, b) { wg("ad_storage") ? Yj() ? a() : Cg(a, "ad_storage") : b ? gg("TAGGING", 3) : Bg(function() { Zj(a, !0) }, ["ad_storage"]) },
        bk = function(a) { return ak(a).map(function(b) { return b.Da }) },
        ak = function(a) {
            var b = [];
            if (!Ji(m) || !H.cookie) return b;
            var c = Mi(a, H.cookie, void 0, "ad_storage");
            if (!c || 0 == c.length) return b;
            for (var d = {}, e = 0; e < c.length; d = { ud: d.ud }, e++) { var f = ck(c[e]); if (null != f) { var g = f,
                        h = g.version;
                    d.ud = g.Da; var k = g.timestamp,
                        n = g.labels,
                        p = Ia(b, function(q) { return function(r) { return r.Da === q.ud } }(d));
                    p ? (p.timestamp = Math.max(p.timestamp, k), p.labels = dk(p.labels, n || [])) : b.push({ version: h, Da: d.ud, timestamp: k, labels: n }) } }
            b.sort(function(q, r) { return r.timestamp - q.timestamp });
            return ek(b)
        };

    function dk(a, b) { for (var c = {}, d = [], e = 0; e < a.length; e++) c[a[e]] = !0, d.push(a[e]); for (var f = 0; f < b.length; f++) c[b[f]] || d.push(b[f]); return d }

    function fk(a) { return a && "string" == typeof a && a.match(Vj) ? a : "_gcl" }
    var hk = function() { var a = ki(m.location.href),
                b = ii(a, "query", !1, void 0, "gclid"),
                c = ii(a, "query", !1, void 0, "gclsrc"),
                d = ii(a, "query", !1, void 0, "wbraid"),
                e = ii(a, "query", !1, void 0, "dclid"); if (!b || !c || !d) { var f = a.hash.replace("#", "");
                b = b || fi(f, "gclid", !1, void 0);
                c = c || fi(f, "gclsrc", !1, void 0);
                d = d || fi(f, "wbraid", !1, void 0) } return gk(b, c, e, d) },
        gk = function(a, b, c, d) {
            var e = {},
                f = function(g, h) { e[h] || (e[h] = []);
                    e[h].push(g) };
            e.gclid = a;
            e.gclsrc = b;
            e.dclid = c;
            void 0 !== d && Wj.test(d) && (e.gbraid = d, f(d, "gb"));
            if (void 0 !==
                a && a.match(Wj)) switch (b) {
                case void 0:
                    f(a, "aw"); break;
                case "aw.ds":
                    f(a, "aw");
                    f(a, "dc"); break;
                case "ds":
                    f(a, "dc"); break;
                case "3p.ds":
                    f(a, "dc"); break;
                case "gf":
                    f(a, "gf"); break;
                case "ha":
                    f(a, "ha") }
            c && f(c, "dc");
            return e
        },
        jk = function(a) { var b = hk();
            Zj(function() { ik(b, !1, a) }) };

    function ik(a, b, c, d, e) {
        function f(x, y) { var w = kk(x, g);
            w && (Vi(w, y, h), k = !0) }
        c = c || {};
        e = e || [];
        var g = fk(c.prefix);
        d = d || Ua();
        var h = dj(c, d, !0);
        h.jb = "ad_storage";
        var k = !1,
            n = Math.round(d / 1E3),
            p = function(x) { var y = ["GCL", n, x];
                0 < e.length && y.push(e.join(".")); return y.join(".") };
        a.aw && f("aw", p(a.aw[0]));
        a.dc && f("dc", p(a.dc[0]));
        a.gf && f("gf", p(a.gf[0]));
        a.ha && f("ha", p(a.ha[0]));
        a.gp && f("gp", p(a.gp[0]));
        if ((void 0 == Uj.enable_gbraid_cookie_write ? 0 : Uj.enable_gbraid_cookie_write) && !k && a.gb) {
            var q = a.gb[0],
                r = kk("gb",
                    g),
                u = !1;
            if (!b)
                for (var t = ak(r), v = 0; v < t.length; v++) t[v].Da === q && t[v].labels && 0 < t[v].labels.length && (u = !0);
            u || f("gb", p(q))
        }
    }
    var mk = function(a, b) { var c = Kj(!0);
            Zj(function() { for (var d = fk(b.prefix), e = 0; e < a.length; ++e) { var f = a[e]; if (void 0 !== Xj[f]) { var g = kk(f, d),
                            h = c[g]; if (h) { var k = Math.min(lk(h), Ua()),
                                n;
                            b: { var p = k; if (Ji(m))
                                    for (var q = Mi(g, H.cookie, void 0, "ad_storage"), r = 0; r < q.length; ++r)
                                        if (lk(q[r]) > p) { n = !0; break b }
                                n = !1 }
                            if (!n) { var u = dj(b, k, !0);
                                u.jb = "ad_storage";
                                Vi(g, h, u) } } } }
                ik(gk(c.gclid, c.gclsrc), !1, b) }) },
        kk = function(a, b) { var c = Xj[a]; if (void 0 !== c) return b + c },
        lk = function(a) {
            return 0 !== nk(a.split(".")).length ? 1E3 * (Number(a.split(".")[1]) ||
                0) : 0
        };

    function ck(a) { var b = nk(a.split(".")); return 0 === b.length ? null : { version: b[0], Da: b[2], timestamp: 1E3 * (Number(b[1]) || 0), labels: b.slice(3) } }

    function nk(a) { return 3 > a.length || "GCL" !== a[0] && "1" !== a[0] || !/^\d+$/.test(a[1]) || !Wj.test(a[2]) ? [] : a }
    var ok = function(a, b, c, d, e) { if (Ha(b) && Ji(m)) { var f = fk(e),
                    g = function() { for (var h = {}, k = 0; k < a.length; ++k) { var n = kk(a[k], f); if (n) { var p = Mi(n, H.cookie, void 0, "ad_storage");
                                p.length && (h[n] = p.sort()[p.length - 1]) } } return h };
                Zj(function() { Qj(g, b, c, d) }) } },
        ek = function(a) { return a.filter(function(b) { return Wj.test(b.Da) }) },
        pk = function(a, b) {
            if (Ji(m)) {
                for (var c = fk(b.prefix), d = {}, e = 0; e < a.length; e++) Xj[a[e]] && (d[a[e]] = Xj[a[e]]);
                Zj(function() {
                    Ma(d, function(f, g) {
                        var h = Mi(c + g, H.cookie, void 0, "ad_storage");
                        h.sort(function(u,
                            t) { return lk(t) - lk(u) });
                        if (h.length) { var k = h[0],
                                n = lk(k),
                                p = 0 !== nk(k.split(".")).length ? k.split(".").slice(3) : [],
                                q = {},
                                r;
                            r = 0 !== nk(k.split(".")).length ? k.split(".")[2] : void 0;
                            q[f] = [r];
                            ik(q, !0, b, n, p) }
                    })
                })
            }
        };

    function qk(a, b) { for (var c = 0; c < b.length; ++c)
            if (a[b[c]]) return !0;
        return !1 }
    var rk = function(a) {
        function b(e, f, g) { g && (e[f] = g) } if (xg()) { var c = hk(); if (qk(c, a)) { var d = {};
                b(d, "gclid", c.gclid);
                b(d, "dclid", c.dclid);
                b(d, "gclsrc", c.gclsrc);
                b(d, "wbraid", c.gbraid);
                Rj(function() { return d }, 3);
                Rj(function() { var e = {}; return e._up = "1", e }, 1) } } };

    function sk(a, b) { var c = fk(b),
            d = kk(a, c); if (!d) return 0; for (var e = ak(d), f = 0, g = 0; g < e.length; g++) f = Math.max(f, e[g].timestamp); return f }

    function tk(a) { var b = 0,
            c; for (c in a)
            for (var d = a[c], e = 0; e < d.length; e++) b = Math.max(b, Number(d[e].timestamp)); return b };
    var Nk = new RegExp(/^(.*\.)?(google|youtube|blogger|withgoogle)(\.com?)?(\.[a-z]{2})?\.?$/),
        Ok = { cl: ["ecl"], customPixels: ["nonGooglePixels"], ecl: ["cl"], ehl: ["hl"], hl: ["ehl"], html: ["customScripts", "customPixels", "nonGooglePixels", "nonGoogleScripts", "nonGoogleIframes"], customScripts: ["html", "customPixels", "nonGooglePixels", "nonGoogleScripts", "nonGoogleIframes"], nonGooglePixels: [], nonGoogleScripts: ["nonGooglePixels"], nonGoogleIframes: ["nonGooglePixels"] },
        dl = {
            cl: ["ecl"],
            customPixels: ["customScripts", "html"],
            ecl: ["cl"],
            ehl: ["hl"],
            hl: ["ehl"],
            html: ["customScripts"],
            customScripts: ["html"],
            nonGooglePixels: ["customPixels", "customScripts", "html", "nonGoogleScripts", "nonGoogleIframes"],
            nonGoogleScripts: ["customScripts", "html"],
            nonGoogleIframes: ["customScripts", "html", "nonGoogleScripts"]
        },
        el = "google customPixels customScripts html nonGooglePixels nonGoogleScripts nonGoogleIframes".split(" ");
    var fl = function() { var a = !1;
            a = !0; return a },
        hl = function(a) {
            var b = Mh("gtm.allowlist") || Mh("gtm.whitelist");
            b && jg(9);
            fl() && (b = "google gtagfl lcl zone oid op".split(" "));
            var c = b && Za(Qa(b), Ok),
                d = Mh("gtm.blocklist") ||
                Mh("gtm.blacklist");
            d || (d = Mh("tagTypeBlacklist")) && jg(3);
            d ? jg(8) : d = [];
            gl() && (d = Qa(d), d.push("nonGooglePixels", "nonGoogleScripts", "sandboxedScripts"));
            0 <= Qa(d).indexOf("google") && jg(2);
            var e = d && Za(Qa(d), dl),
                f = {};
            return function(g) {
                var h = g && g[Ld.Eb];
                if (!h || "string" != typeof h) return !0;
                h = h.replace(/^_*/, "");
                if (void 0 !== f[h]) return f[h];
                var k = Fh[h] || [],
                    n = a(h, k);
                if (b) {
                    var p;
                    if (p = n) a: { if (0 > c.indexOf(h))
                            if (k && 0 < k.length)
                                for (var q = 0; q < k.length; q++) { if (0 > c.indexOf(k[q])) { jg(11);
                                        p = !1; break a } } else { p = !1; break a }
                            p = !0 }
                    n = p
                }
                var r = !1;
                if (d) { var u = 0 <= e.indexOf(h); if (u) r = u;
                    else { var t = La(e, k || []);
                        t && jg(10);
                        r = t } }
                var v = !n || r;
                v || !(0 <= k.indexOf("sandboxedScripts")) || c && -1 !== c.indexOf("sandboxedScripts") || (v = La(e, el));
                return f[h] = v
            }
        },
        gl = function() { return Nk.test(m.location && m.location.hostname) };
    var il = !1,
        jl = 0,
        kl = [];

    function ll(a) { if (!il) { var b = H.createEventObject,
                c = "complete" == H.readyState,
                d = "interactive" == H.readyState; if (!a || "readystatechange" != a.type || c || !b && d) { il = !0; for (var e = 0; e < kl.length; e++) I(kl[e]) }
            kl.push = function() { for (var f = 0; f < arguments.length; f++) I(arguments[f]); return 0 } } }

    function ml() { if (!il && 140 > jl) { jl++; try { H.documentElement.doScroll("left"), ll() } catch (a) { m.setTimeout(ml, 50) } } }
    var nl = function(a) { il ? a() : kl.push(a) };
    var ql = function(a, b) { this.g = !1;
            this.C = [];
            this.I = { tags: [] };
            this.P = !1;
            this.o = this.s = 0;
            pl(this, a, b) },
        rl = function(a, b, c, d) { if (wh.hasOwnProperty(b) || "__zone" === b) return -1; var e = {};
            oc(d) && (e = pc(d, e));
            e.id = c;
            e.status = "timeout"; return a.I.tags.push(e) - 1 },
        sl = function(a, b, c, d) { var e = a.I.tags[b];
            e && (e.status = c, e.executionTime = d) },
        tl = function(a) { if (!a.g) { for (var b = a.C, c = 0; c < b.length; c++) b[c]();
                a.g = !0;
                a.C.length = 0 } },
        pl = function(a, b, c) { Ba(b) && a.Sd(b);
            c && m.setTimeout(function() { return tl(a) }, Number(c)) };
    ql.prototype.Sd = function(a) { var b = this,
            c = Wa(function() { return I(function() { a(J.J, b.I) }) });
        this.g ? c() : this.C.push(c) };
    var ul = function(a) { a.s++; return Wa(function() { a.o++;
            a.P && a.o >= a.s && tl(a) }) };
    var vl = function() {
            function a(d) { return !Fa(d) || 0 > d ? 0 : d } if (!T._li && m.performance && m.performance.timing) { var b = m.performance.timing.navigationStart,
                    c = Fa(Nh.get("gtm.start")) ? Nh.get("gtm.start") : 0;
                T._li = { cst: a(c - b), cbt: a(Ch - b) } } },
        wl = function(a) { m.performance && m.performance.mark(J.J + "_" + a + "_start") },
        xl = function(a) {
            if (m.performance) {
                var b = J.J + "_" + a + "_start",
                    c = J.J + "_" + a + "_duration";
                m.performance.measure(c, b);
                var d = m.performance.getEntriesByName(c)[0];
                m.performance.clearMarks(b);
                m.performance.clearMeasures(c);
                var e = T._p || {};
                void 0 === e[a] && (e[a] = d.duration, T._p = e);
                return d.duration
            }
        },
        yl = function() { if (m.performance && m.performance.now) { var a = T._p || {};
                a.PAGEVIEW = m.performance.now();
                T._p = a } };
    var zl = {},
        Al = function() { return m.GoogleAnalyticsObject && m[m.GoogleAnalyticsObject] },
        Bl = !1;

    function El() { return m.GoogleAnalyticsObject || "ga" }
    var Fl = function(a) {},
        Gl = function(a, b) {
            return function() {
                var c = Al(),
                    d = c && c.getByName && c.getByName(a);
                if (d) {
                    var e = d.get("sendHitTask");
                    d.set("sendHitTask", function(f) {
                        var g = f.get("hitPayload"),
                            h = f.get("hitCallback"),
                            k = 0 > g.indexOf("&tid=" + b);
                        k && (f.set("hitPayload", g.replace(/&tid=UA-[0-9]+-[0-9]+/, "&tid=" + b), !0), f.set("hitCallback", void 0, !0));
                        e(f);
                        k && (f.set("hitPayload",
                            g, !0), f.set("hitCallback", h, !0), f.set("_x_19", void 0, !0), e(f))
                    })
                }
            }
        };
    var Nl = function(a) {},
        Rl = function(a) {},
        Sl =
        function() { return "&tc=" + le.filter(function(a) { return a }).length },
        Vl = function() { 2022 <= Tl().length && Ul() },
        Wl = function(a) { return 0 === a.indexOf("gtm.") ? encodeURIComponent(a) : "*" },
        Yl = function() { Xl || (Xl = m.setTimeout(Ul, 500)) },
        Ul = function() {
            Xl && (m.clearTimeout(Xl), Xl = void 0);
            if (void 0 !== Zl && (!$l[Zl] || am || bm))
                if (cm[Zl] || dm.Dj() || 0 >= em--) jg(1), cm[Zl] = !0;
                else {
                    dm.Zj();
                    var a = Tl(!0);
                    Rb(a);
                    $l[Zl] = !0;
                    gm = fm = im = jm = km = bm = am = "";
                    hm = []
                }
        },
        Tl = function(a) { var b = Zl; if (void 0 === b) return ""; var c = hg("GTM"),
                d = hg("TAGGING"); return [lm, $l[b] ? "" : "&es=1", mm[b], Nl(b), c ? "&u=" + c : "", d ? "&ut=" + d : "", Sl(), am, bm, km, jm, Rl(a), im, fm, gm ? "&dl=" + encodeURIComponent(gm) : "", 0 < hm.length ? "&tdp=" + hm.join(".") : "", "&z=0"].join("") },
        om = function() { lm = nm() },
        nm = function() { return [Dh, "&v=3&t=t", "&pid=" + Ja(), "&rv=" + J.Pd].join("") },
        Ql = ["L", "S", "Y"],
        Ml = ["S", "E"],
        pm = {
            sampleRate: "0.005000",
            Lh: "",
            Kh: Number("5")
        },
        qm = 0 <= H.location.search.indexOf("?gtm_latency=") || 0 <= H.location.search.indexOf("&gtm_latency="),
        rm;
    if (!(rm = qm)) { var sm = Math.random(),
            tm = pm.sampleRate;
        rm = sm < tm }
    var um = rm,
        vm = { label: J.J + " Container", children: [{ label: "Initialization", children: [] }] },
        lm = nm(),
        $l = {},
        am = "",
        bm = "",
        im = "",
        jm = "",
        fm = "",
        hm = [],
        gm = "",
        Pl = {},
        Ol = !1,
        Ll = {},
        wm = {},
        km = "",
        Zl = void 0,
        mm = {},
        cm = {},
        Xl = void 0,
        xm = 5;
    0 < pm.Kh && (xm = pm.Kh);
    var dm = function(a, b) { for (var c = 0, d = [], e = 0; e < a; ++e) d.push(0); return { Dj: function() { return c < a ? !1 : Ua() - d[c % a] < b }, Zj: function() { var f = c++ % a;
                    d[f] = Ua() } } }(xm, 1E3),
        em = 1E3,
        zm = function(a, b) { if (um && !cm[a] && Zl !== a) { Ul();
                Zl = a;
                im = am = "";
                mm[a] = "&e=" + Wl(b) + "&eid=" + a;
                Yl(); } },
        Am = function(a, b, c, d) {
            if (um && b) {
                var e, f = String(b[Ld.Eb] || "").replace(/_/g, "");
                0 === f.indexOf("cvt") && (f = "cvt");
                e = f;
                var g = c + e;
                if (!cm[a]) {
                    a !==
                        Zl && (Ul(), Zl = a);
                    am = am ? am + "." + g : "&tr=" + g;
                    var h = b["function"];
                    if (!h) throw Error("Error: No function name given for function call.");
                    var k = (ne[h] ? "1" : "2") + e;
                    im = im ? im + "." + k : "&ti=" + k;
                    Yl();
                    Vl()
                }
            }
        };
    var Hm =
        function(a, b, c) { if (um && !cm[a]) { a !== Zl && (Ul(), Zl = a); var d = c + b;
                bm = bm ? bm + "." + d : "&epr=" + d;
                Yl();
                Vl() } },
        Im = function(a, b, c) {};

    function Jm(a, b, c, d) { var e = le[a],
            f = Km(a, b, c, d); if (!f) return null; var g = te(e[Ld.Hg], c, []); if (g && g.length) { var h = g[0];
            f = Jm(h.index, { onSuccess: f, onFailure: 1 === h.eh ? b.terminate : f, terminate: b.terminate }, c, d) } return f }

    function Km(a, b, c, d) {
        function e() {
            if (f[Ld.Gi]) h();
            else {
                var x = ue(f, c, []);
                var y = x[Ld.Qh];
                if (null != y)
                    for (var w = 0; w < y.length; w++)
                        if (!Kg(y[w])) { h(); return }
                var A = rl(c.Hb, String(f[Ld.Eb]), Number(f[Ld.Jg]), x[Ld.Hi]),
                    B = !1;
                x.vtp_gtmOnSuccess = function() {
                    if (!B) {
                        B = !0;
                        var E = Ua() - D;
                        Am(c.id, le[a], "5", E);
                        sl(c.Hb, A, "success",
                            E);
                        g()
                    }
                };
                x.vtp_gtmOnFailure = function() { if (!B) { B = !0; var E = Ua() - D;
                        Am(c.id, le[a], "6", E);
                        sl(c.Hb, A, "failure", E);
                        h() } };
                x.vtp_gtmTagId = f.tag_id;
                x.vtp_gtmEventId = c.id;
                Am(c.id, f, "1");
                var C = function() { var E = Ua() - D;
                    Am(c.id, f, "7", E);
                    sl(c.Hb, A, "exception", E);
                    B || (B = !0, h()) };
                var D = Ua();
                try { se(x, { event: c, index: a, type: 1 }) } catch (E) { C(E) }
            }
        }
        var f = le[a],
            g = b.onSuccess,
            h = b.onFailure,
            k = b.terminate;
        if (c.zf(f)) return null;
        var n = te(f[Ld.Kg], c, []);
        if (n && n.length) { var p = n[0],
                q = Jm(p.index, { onSuccess: g, onFailure: h, terminate: k }, c, d); if (!q) return null;
            g = q;
            h = 2 === p.eh ? k : q }
        if (f[Ld.Bg] || f[Ld.Li]) {
            var r =
                f[Ld.Bg] ? me : c.nk,
                u = g,
                t = h;
            if (!r[a]) { e = Wa(e); var v = Lm(a, r, e);
                g = v.onSuccess;
                h = v.onFailure }
            return function() { r[a](u, t) }
        }
        return e
    }

    function Lm(a, b, c) { var d = [],
            e = [];
        b[a] = Mm(d, e, c); return { onSuccess: function() { b[a] = Nm; for (var f = 0; f < d.length; f++) d[f]() }, onFailure: function() { b[a] = Om; for (var f = 0; f < e.length; f++) e[f]() } } }

    function Mm(a, b, c) { return function(d, e) { a.push(d);
            b.push(e);
            c() } }

    function Nm(a) { a() }

    function Om(a, b) { b() };

    function Pm(a, b) { if (a) { var c = "" + a;
            0 !== c.indexOf("http://") && 0 !== c.indexOf("https://") && (c = "https://" + c); "/" === c[c.length - 1] && (c = c.substring(0, c.length - 1)); return ki("" + c + b).href } }

    function Qm(a, b) { return Rm() ? Pm(a, b) : void 0 }

    function Rm() { var a = !1; return a }

    function Sm() { return !!J.Vc && "SGTM_TOKEN" !== J.Vc.replaceAll("@@", "") };
    var Tm = function() { var a = !1; return a };
    var Vm = function(a, b, c, d) { return (2 === Um() || d || "http:" != m.location.protocol ? a : b) + c },
        Um = function() { var a = Pb(),
                b; if (1 === a) a: { var c = yh;c = c.toLowerCase(); for (var d = "https://" + c, e = "http://" + c, f = 1, g = H.getElementsByTagName("script"), h = 0; h < g.length && 100 > h; h++) { var k = g[h].src; if (k) { k = k.toLowerCase(); if (0 === k.indexOf(e)) { b = 3; break a }
                        1 === f && 0 === k.indexOf(d) && (f = 2) } }
                b = f }
            else b = a; return b };
    var Wm = !1;
    var Ym = { initialized: 11, complete: 12, interactive: 13 },
        Zm = {},
        $m = Object.freeze((Zm[S.Db] = !0, Zm[S.mc] = !0, Zm)),
        an = {},
        bn = Object.freeze((an[S.Aa] = !0, an)),
        cn = {},
        dn = 0 <= H.location.search.indexOf("?gtm_diagnostics=") || 0 <= H.location.search.indexOf("&gtm_diagnostics="),
        fn = function(a, b, c) {},
        gn = function(a) {};

    function hn(a, b) { var c = {},
            d; for (d in b) b.hasOwnProperty(d) && (c[d] = !0); for (var e in a) a.hasOwnProperty(e) && (c[e] = !0); return c }

    function en(a, b, c, d) { c = void 0 === c ? {} : c;
        d = void 0 === d ? "" : d; if (a === b) return []; var e = function(q, r) { var u = r[q]; return void 0 === u ? bn[q] : u },
            f; for (f in hn(a, b))
            if (!$m[f]) { var g = (d ? d + "." : "") + f,
                    h = e(f, a),
                    k = e(f, b),
                    n = "object" === mc(h) || "array" === mc(h),
                    p = "object" === mc(k) || "array" === mc(k); if (n && p) en(h, k, c, g);
                else if (n || p || h !== k) c[g] = !0 }
        return Object.keys(c) }
    var jn = function() { this.eventModel = {};
            this.targetConfig = {};
            this.containerConfig = {};
            this.globalConfig = {};
            this.dataLayerConfig = null;
            this.remoteConfig = {};
            this.onSuccess = function() {};
            this.onFailure = function() {};
            this.setContainerTypeLoaded = function() {};
            this.getContainerTypeLoaded = function() {};
            this.eventId = void 0;
            this.isGtmEvent = !1 },
        kn = function(a) { var b = new jn;
            b.eventModel = a; return b },
        ln = function(a, b) { a.targetConfig = b; return a },
        mn = function(a, b) { a.containerConfig = b; return a },
        nn = function(a, b) {
            a.globalConfig =
                b;
            return a
        },
        on = function(a, b) { a.dataLayerConfig = b; return a },
        pn = function(a, b) { a.remoteConfig = b; return a },
        qn = function(a, b) { a.onSuccess = b; return a },
        rn = function(a, b) { a.setContainerTypeLoaded = b; return a },
        sn = function(a, b) { a.getContainerTypeLoaded = b; return a },
        tn = function(a, b) { a.onFailure = b; return a };
    l = jn.prototype;
    l.getWithConfig = function(a) { if (void 0 !== this.eventModel[a]) return this.eventModel[a]; if (void 0 !== this.targetConfig[a]) return this.targetConfig[a]; if (void 0 !== this.containerConfig[a]) return this.containerConfig[a]; if (void 0 !== this.globalConfig[a]) return this.globalConfig[a]; if (void 0 !== this.remoteConfig[a]) return this.remoteConfig[a] };
    l.getTopLevelKeys = function() {
        function a(c) { for (var d = Object.keys(c), e = 0; e < d.length; ++e) b[d[e]] = 1 } var b = {};
        a(this.eventModel);
        a(this.targetConfig);
        a(this.containerConfig);
        a(this.globalConfig); return Object.keys(b) };
    l.getMergedValues = function(a, b) {
        function c(f) { oc(f) && Ma(f, function(g, h) { e = !0;
                d[g] = h }) } var d = {},
            e = !1;
        b && 1 !== b || (c(this.remoteConfig[a]), c(this.globalConfig[a]), c(this.containerConfig[a]), c(this.targetConfig[a]));
        b && 2 !== b || c(this.eventModel[a]); return e ? d : void 0 };
    l.getKeysFromFirstOfAnyScope = function(a) { var b = {},
            c = !1,
            d = function(e) { for (var f = 0; f < a.length; f++) void 0 !== e[a[f]] && (b[a[f]] = e[a[f]], c = !0); return c }; if (d(this.eventModel) || d(this.targetConfig) || d(this.containerConfig) || d(this.globalConfig)) return b;
        d(this.remoteConfig); return b };
    l.getEventModelKeys = function() { var a = [],
            b; for (b in this.eventModel) b !== S.Db && this.eventModel.hasOwnProperty(b) && void 0 !== this.eventModel[b] && a.push(b); return a };

    function un() { T.dedupe_gclid || (T.dedupe_gclid = "" + Zi()); return T.dedupe_gclid };
    var vn;
    if (3 === J.Pd.length) vn = "g";
    else { var wn = "G";
        wn = "g";
        vn = wn }
    var xn = { "": "n", UA: "u", AW: "a", DC: "d", G: "e", GF: "f", HA: "h", GTM: vn, OPT: "o" },
        yn = function(a) { var b = J.J.split("-"),
                c = b[0].toUpperCase(),
                d = xn[c] || "i",
                e = a && "GTM" === c ? b[1] : "OPT" === c ? b[1] : "",
                f; if (3 === J.Pd.length) { var g = "w";
                g = Tm() ? "s" : "o";
                f = "2" + g } else f = ""; return f + d + J.Pd + e };

    function zn(a, b) { if ("" === a) return b; var c = Number(a); return isNaN(c) ? b : c };
    var An = function(a, b) { a.addEventListener && a.addEventListener.call(a, "message", b, !1) };

    function Bn() { return vb("iPhone") && !vb("iPod") && !vb("iPad") };
    vb("Opera");
    vb("Trident") || vb("MSIE");
    vb("Edge");
    !vb("Gecko") || -1 != ub().toLowerCase().indexOf("webkit") && !vb("Edge") || vb("Trident") || vb("MSIE") || vb("Edge"); - 1 != ub().toLowerCase().indexOf("webkit") && !vb("Edge") && vb("Mobile");
    vb("Macintosh");
    vb("Windows");
    vb("Linux") || vb("CrOS");
    var Cn = ma.navigator || null;
    Cn && (Cn.appVersion || "").indexOf("X11");
    vb("Android");
    Bn();
    vb("iPad");
    vb("iPod");
    Bn() || vb("iPad") || vb("iPod");
    ub().toLowerCase().indexOf("kaios");
    var Dn = function(a) { if (!a || !H.head) return null; var b, c;
        c = void 0 === c ? document : c;
        b = c.createElement("meta");
        H.head.appendChild(b);
        b.httpEquiv = "origin-trial";
        b.content = a; return b };
    var En = function() {};
    var Fn = function(a) { void 0 !== a.addtlConsent && "string" !== typeof a.addtlConsent && (a.addtlConsent = void 0);
            void 0 !== a.gdprApplies && "boolean" !== typeof a.gdprApplies && (a.gdprApplies = void 0); return void 0 !== a.tcString && "string" !== typeof a.tcString || void 0 !== a.listenerId && "number" !== typeof a.listenerId ? 2 : a.cmpStatus && "error" !== a.cmpStatus ? 0 : 3 },
        Gn = function(a, b) { this.o = a;
            this.g = null;
            this.C = {};
            this.P = 0;
            this.I = void 0 === b ? 500 : b;
            this.s = null };
    la(Gn, En);
    Gn.prototype.addEventListener = function(a) {
        var b = {},
            c = Ii(function() { return a(b) }),
            d = 0; - 1 !== this.I && (d = setTimeout(function() { b.tcString = "tcunavailable";
            b.internalErrorState = 1;
            c() }, this.I));
        var e = function(f, g) { clearTimeout(d);
            f ? (b = f, b.internalErrorState = Fn(b), g && 0 === b.internalErrorState || (b.tcString = "tcunavailable", g || (b.internalErrorState = 3))) : (b.tcString = "tcunavailable", b.internalErrorState = 3);
            a(b) };
        try { Hn(this, "addEventListener", e) } catch (f) {
            b.tcString = "tcunavailable", b.internalErrorState = 3, d && (clearTimeout(d),
                d = 0), c()
        }
    };
    Gn.prototype.removeEventListener = function(a) { a && a.listenerId && Hn(this, "removeEventListener", null, a.listenerId) };
    var Jn = function(a, b, c) {
            var d;
            d = void 0 === d ? "755" : d;
            var e;
            a: { if (a.publisher && a.publisher.restrictions) { var f = a.publisher.restrictions[b]; if (void 0 !== f) { e = f[void 0 === d ? "755" : d]; break a } }
                e = void 0 }
            var g = e;
            if (0 === g) return !1;
            var h = c;
            2 === c ? (h = 0, 2 === g && (h = 1)) : 3 === c && (h = 1, 1 === g && (h = 0));
            var k;
            if (0 === h)
                if (a.purpose && a.vendor) { var n = In(a.vendor.consents, void 0 === d ? "755" : d);
                    k = n && "1" === b && a.purposeOneTreatment && "CH" === a.publisherCC ? !0 : n && In(a.purpose.consents, b) } else k = !0;
            else k = 1 === h ? a.purpose && a.vendor ? In(a.purpose.legitimateInterests,
                b) && In(a.vendor.legitimateInterests, void 0 === d ? "755" : d) : !0 : !0;
            return k
        },
        In = function(a, b) { return !(!a || !a[b]) },
        Hn = function(a, b, c, d) { c || (c = function() {}); if ("function" === typeof a.o.__tcfapi) { var e = a.o.__tcfapi;
                e(b, 2, c, d) } else if (Kn(a)) { Ln(a); var f = ++a.P;
                a.C[f] = c; if (a.g) { var g = {};
                    a.g.postMessage((g.__tcfapiCall = { command: b, version: 2, callId: f, parameter: d }, g), "*") } } else c({}, !1) },
        Kn = function(a) {
            if (a.g) return a.g;
            var b;
            a: {
                for (var c = a.o, d = 0; 50 > d; ++d) {
                    var e;
                    try { e = !(!c.frames || !c.frames.__tcfapiLocator) } catch (h) {
                        e = !1
                    }
                    if (e) { b = c; break a }
                    var f;
                    b: { try { var g = c.parent; if (g && g != c) { f = g; break b } } catch (h) {}
                        f = null }
                    if (!(c = f)) break
                }
                b = null
            }
            a.g = b;
            return a.g
        },
        Ln = function(a) { a.s || (a.s = function(b) { try { var c;
                    c = ("string" === typeof b.data ? JSON.parse(b.data) : b.data).__tcfapiReturn;
                    a.C[c.callId](c.returnValue, c.success) } catch (d) {} }, An(a.o, a.s)) };
    var Mn = !0;
    Mn = !1;
    var Nn = { 1: 0, 3: 0, 4: 0, 7: 3, 9: 3, 10: 3 },
        On = zn("", 550),
        Pn = zn("", 500);

    function Qn() { var a = T.tcf || {}; return T.tcf = a }
    var Vn = function() {
        var a = Qn(),
            b = new Gn(m, Mn ? 3E3 : -1);
        if (!0 === m.gtag_enable_tcf_support && !a.active && ("function" === typeof m.__tcfapi || "function" === typeof b.o.__tcfapi || null != Kn(b))) {
            a.active = !0;
            a.md = {};
            Rn();
            var c = null;
            Mn ? c = m.setTimeout(function() { Sn(a);
                Tn(a);
                c = null }, Pn) : a.tcString = "tcunavailable";
            try {
                b.addEventListener(function(d) {
                    c && (clearTimeout(c), c = null);
                    if (0 !== d.internalErrorState) Sn(a), Tn(a);
                    else {
                        var e;
                        a.gdprApplies = d.gdprApplies;
                        if (!1 === d.gdprApplies) e = Un(), b.removeEventListener(d);
                        else if ("tcloaded" ===
                            d.eventStatus || "useractioncomplete" === d.eventStatus || "cmpuishown" === d.eventStatus) {
                            var f = {},
                                g;
                            for (g in Nn)
                                if (Nn.hasOwnProperty(g))
                                    if ("1" === g) {
                                        var h = d,
                                            k = !0;
                                        k = void 0 === k ? !1 : k;
                                        var n;
                                        var p = h;
                                        !1 === p.gdprApplies ? n = !0 : (void 0 === p.internalErrorState && (p.internalErrorState = Fn(p)), n = "error" === p.cmpStatus || 0 !== p.internalErrorState || "loaded" === p.cmpStatus && ("tcloaded" === p.eventStatus || "useractioncomplete" === p.eventStatus) ? !0 : !1);
                                        f["1"] = n ? !1 === h.gdprApplies || "tcunavailable" === h.tcString || void 0 === h.gdprApplies &&
                                            !k || "string" !== typeof h.tcString || !h.tcString.length ? !0 : Jn(h, "1", 0) : !1
                                    } else f[g] = Jn(d, g, Nn[g]);
                            e = f
                        }
                        e && (a.tcString = d.tcString || "tcempty", a.md = e, Tn(a))
                    }
                })
            } catch (d) { c && (clearTimeout(c), c = null), Sn(a), Tn(a) }
        }
    };

    function Sn(a) { a.type = "e";
        a.tcString = "tcunavailable";
        Mn && (a.md = Un()) }

    function Rn() { var a = {},
            b = (a.ad_storage = "denied", a.wait_for_update = On, a);
        Hg(b) }

    function Un() { var a = {},
            b; for (b in Nn) Nn.hasOwnProperty(b) && (a[b] = !0); return a }

    function Tn(a) { var b = {},
            c = (b.ad_storage = a.md["1"] ? "granted" : "denied", b);
        Jg(c, 0, { gdprApplies: a ? a.gdprApplies : void 0, tcString: Wn() }) }
    var Wn = function() { var a = Qn(); return a.active ? a.tcString || "" : "" },
        Xn = function() { var a = Qn(); return a.active && void 0 !== a.gdprApplies ? a.gdprApplies ? "1" : "0" : "" },
        Yn = function(a) { if (!Nn.hasOwnProperty(String(a))) return !0; var b = Qn(); return b.active && b.md ? !!b.md[String(a)] : !0 };
    var fo = !1;
    var go = function() { this.g = {} },
        ho = function(a, b, c) { null != c && (a.g[b] = c) },
        io = function(a) { return Object.keys(a.g).map(function(b) { return encodeURIComponent(b) + "=" + encodeURIComponent(a.g[b]) }).join("&") },
        ko = function(a, b, c, d, e) {};
    var mo = /[A-Z]+/,
        no = /\s/,
        oo = function(a) { if (Da(a)) { a = Ra(a); var b = a.indexOf("-"); if (!(0 > b)) { var c = a.substring(0, b); if (mo.test(c)) { for (var d = a.substring(b + 1).split("/"), e = 0; e < d.length; e++)
                            if (!d[e] || no.test(d[e]) && ("AW" !== c || 1 !== e)) return;
                        return { id: a, prefix: c, containerId: c + "-" + d[0], O: d } } } } },
        qo = function(a) { for (var b = {}, c = 0; c < a.length; ++c) { var d = oo(a[c]);
                d && (b[d.id] = d) }
            po(b); var e = [];
            Ma(b, function(f, g) { e.push(g) }); return e };

    function po(a) { var b = [],
            c; for (c in a)
            if (a.hasOwnProperty(c)) { var d = a[c]; "AW" === d.prefix && d.O[1] && b.push(d.containerId) }
        for (var e = 0; e < b.length; ++e) delete a[b[e]] };
    var Jo = !1;

    function Ko() { if (Ba(Hb.joinAdInterestGroup)) return !0;
        Jo || (Dn(''), Jo = !0); return Ba(Hb.joinAdInterestGroup) }

    function Lo(a, b) { var c = void 0; try { c = H.querySelector('iframe[data-tagging-id="' + b + '"]') } catch (e) {} if (c) { var d = Number(c.dataset.loadTime); if (d && 6E4 > Ua() - d) { gg("TAGGING", 9); return } } else try { if (50 <= H.querySelectorAll('iframe[allow="join-ad-interest-group"][data-tagging-id*="-"]').length) { gg("TAGGING", 10); return } } catch (e) {}
        Qb(a, void 0, { allow: "join-ad-interest-group" }, { taggingId: b, loadTime: Ua() }, c) };
    var up = function(a, b, c) { this.ra = a;
            this.eventName = b;
            this.H = c;
            this.F = {};
            this.metadata = {};
            this.Wa = !1 },
        vp = function(a, b, c) { var d = a.H.getWithConfig(b);
            void 0 !== d ? a.F[b] = d : void 0 !== c && (a.F[b] = c) },
        wp = function(a, b, c) { var d = Vh(a.ra); return d && d.hasOwnProperty(b) ? d[b] : c };

    function xp(a) { return { getDestinationId: function() { return a.ra }, getEventName: function() { return a.eventName }, setEventName: function(b) { return void(a.eventName = b) }, getHitData: function(b) { return a.F[b] }, setHitData: function(b, c) { return void(a.F[b] = c) }, setHitDataIfNotDefined: function(b, c) { void 0 === a.F[b] && (a.F[b] = c) }, copyToHitData: function(b, c) { vp(a, b, c) }, getMetadata: function(b) { return a.metadata[b] }, setMetadata: function(b, c) { return void(a.metadata[b] = c) }, abort: function() { return void(a.Wa = !0) }, getProcessedEvent: function() { return a } } };
    var zp = function(a) { var b = yp[a.ra]; if (!a.Wa && b)
                for (var c = xp(a), d = 0; d < b.length; ++d) { try { b[d](c) } catch (e) { a.Wa = !0 } if (a.Wa) break } },
        Ap = function(a, b) { var c = yp[a];
            c || (c = yp[a] = []);
            c.push(b) },
        yp = {};
    var Wp = function() { var a = !0;
            Yn(7) && Yn(9) && Yn(10) || (a = !1); return a },
        Xp = function() { var a = !0;
            Yn(3) && Yn(4) || (a = !1); return a };

    function Qq() { return T.gcq = T.gcq || new Rq }
    var Sq = function(a, b, c) { Qq().register(a, b, c) },
        Tq = function(a, b, c, d) { Qq().push("event", [b, a], c, d) },
        Uq = function(a, b, c) { Qq().insert("event", [b, a], c) },
        Vq = function(a, b) { Qq().push("config", [a], b) },
        Wq = function(a, b, c, d) { Qq().push("get", [a, b], c, d) },
        Xq = function(a) { return Qq().getRemoteConfig(a) },
        Yq = {},
        Zq = function() { this.status = 1;
            this.containerConfig = {};
            this.targetConfig = {};
            this.remoteConfig = {};
            this.o = {};
            this.s = null;
            this.claimed = this.g = !1 },
        $q = function(a, b, c, d, e) {
            this.type = a;
            this.s = b;
            this.U = c || "";
            this.g = d;
            this.o =
                e
        },
        Rq = function() { this.o = {};
            this.s = {};
            this.g = [];
            this.C = { AW: !1, UA: !1 } },
        ar = function(a, b) { var c = oo(b); return a.o[c.containerId] = a.o[c.containerId] || new Zq },
        br = function(a, b, c) { if (b) { var d = oo(b); if (d && 1 === ar(a, b).status) { ar(a, b).status = 2; var e = {};
                    um && (e.timeoutId = m.setTimeout(function() { jg(38);
                        Yl() }, 3E3));
                    a.push("require", [e], d.containerId);
                    Yq[d.containerId] = Ua(); if (Tm()) {} else if (Wm) Xm(d.containerId, c, !0);
                    else { var g = "/gtag/js?id=" + encodeURIComponent(d.containerId) + "&l=" + J.Z + "&cx=c";
                        Sm() && (g += "&sign=" + J.Vc); var h = ("http:" != m.location.protocol ? "https:" : "http:") + ("//www.googletagmanager.com" + g),
                            k = Qm(c, g) || h;
                        Ob(k) } } } },
        cr = function(a, b, c, d) {
            if (d.U) {
                var e = ar(a, d.U),
                    f = e.s;
                if (f) {
                    var g = pc(c),
                        h = pc(e.targetConfig[d.U]),
                        k = pc(e.containerConfig),
                        n = pc(e.remoteConfig),
                        p = pc(a.s),
                        q = null;
                    T.mdm && (q = pc(Jh));
                    var r = Mh("gtm.uniqueEventId"),
                        u = oo(d.U).prefix,
                        t = Wa(function() { var x = g[S.bc];
                            x && I(x) }),
                        v = sn(rn(tn(qn(on(nn(pn(mn(ln(kn(g), h), k), n), p), q), function() { Hm(r, u, "2");
                            t() }), function() { Hm(r, u, "3");
                            t() }), function(x, y) { a.C[x] = y }), function(x) { return a.C[x] });
                    try {
                        Hm(r, u, "1"), fn(d.type, d.U, v), "config" ===
                            d.type && gn(d.U);
                        f(d.U, b, d.s, v)
                    } catch (x) { Hm(r, u, "4"); }
                }
            }
        };
    l = Rq.prototype;
    l.register = function(a, b, c) { var d = ar(this, a); if (3 !== d.status) { d.s = b;
            d.status = 3;
            c && (pc(d.remoteConfig, c), d.remoteConfig = c); var e = oo(a),
                f = Yq[e.containerId]; if (void 0 !== f) { var g = T[e.containerId].bootstrap,
                    h = e.prefix.toUpperCase();
                T[e.containerId]._spx && (h = h.toLowerCase()); var k = Mh("gtm.uniqueEventId"),
                    n = h,
                    p = Ua() - g; if (um && !cm[k]) { k !== Zl && (Ul(), Zl = k); var q = n + "." + Math.floor(g - f) + "." + Math.floor(p);
                    jm = jm ? jm + "," + q : "&cl=" + q }
                delete Yq[e.containerId] }
            this.flush() } };
    l.notifyContainerLoaded = function(a, b) { var c = this,
            d = function(f) { if (oo(f)) { var g = ar(c, f);
                    g.status = 3;
                    g.claimed = !0 } };
        d(a); for (var e = 0; e < b.length; e++) d(b[e]);
        this.flush() };
    l.push = function(a, b, c, d) { if (void 0 !== c) { if (!oo(c)) return;
            br(this, c, b[0][S.Ba] || this.s[S.Ba]);
            ar(this, c).g && (d = !1) }
        this.g.push(new $q(a, Math.floor(Ua() / 1E3), c, b, d));
        d || this.flush() };
    l.insert = function(a, b, c) { var d = Math.floor(Ua() / 1E3);
        0 < this.g.length ? this.g.splice(1, 0, new $q(a, d, c, b, !1)) : this.g.push(new $q(a, d, c, b, !1)) };
    l.flush = function(a) {
        for (var b = this, c = [], d = !1, e = {}; this.g.length;) {
            var f = this.g[0];
            if (f.o) !f.U || ar(this, f.U).g ? (f.o = !1, this.g.push(f)) : c.push(f), this.g.shift();
            else {
                var g = void 0;
                switch (f.type) {
                    case "require":
                        g = ar(this, f.U);
                        if (3 !== g.status && !a) { this.g.push.apply(this.g, c); return }
                        um && m.clearTimeout(f.g[0].timeoutId);
                        break;
                    case "set":
                        Ma(f.g[0], function(r, u) { pc(ab(r, u), b.s) });
                        break;
                    case "config":
                        g = ar(this, f.U);
                        if (g.claimed) break;
                        e.$a = {};
                        Ma(f.g[0], function(r) { return function(u, t) { pc(ab(u, t), r.$a) } }(e));
                        var h = !!e.$a[S.Kd];
                        delete e.$a[S.Kd];
                        var k = oo(f.U),
                            n = k.containerId === k.id;
                        h || (n ? g.containerConfig = {} : g.targetConfig[f.U] = {});
                        g.g && h || cr(this, S.Ga, e.$a, f);
                        g.g = !0;
                        delete e.$a[S.Db];
                        n ? pc(e.$a, g.containerConfig) : (pc(e.$a, g.targetConfig[f.U]), jg(70));
                        d = !0;
                        break;
                    case "event":
                        if (g = ar(this,
                                f.U), g.claimed) break;
                        e.td = {};
                        Ma(f.g[0], function(r) { return function(u, t) { pc(ab(u, t), r.td) } }(e));
                        cr(this, f.g[1], e.td, f);
                        break;
                    case "get":
                        if (g = ar(this, f.U), g.claimed) break;
                        var p = {},
                            q = (p[S.eb] = f.g[0], p[S.pb] = f.g[1], p);
                        cr(this, S.Pa, q, f)
                }
                this.g.shift();
                dr(this, f)
            }
            e = { $a: e.$a, td: e.td }
        }
        this.g.push.apply(this.g,
            c);
        d && this.flush()
    };
    var dr = function(a, b) { if ("require" !== b.type)
            if (b.U)
                for (var c = a.getCommandListeners(b.U)[b.type] || [], d = 0; d < c.length; d++) c[d]();
            else
                for (var e in a.o)
                    if (a.o.hasOwnProperty(e)) { var f = a.o[e]; if (f && f.o)
                            for (var g = f.o[b.type] || [], h = 0; h < g.length; h++) g[h]() } };
    Rq.prototype.getRemoteConfig = function(a) { return ar(this, a).remoteConfig };
    Rq.prototype.getCommandListeners = function(a) { return ar(this, a).o };
    var er = !1;
    var fr = !1;
    var gr = {},
        hr = {},
        ir = function(a) { T.addTargetToGroup ? T.addTargetToGroup(a) : (T.pendingDefaultTargets = T.pendingDefaultTargets || [], T.pendingDefaultTargets.push(a)) },
        jr = function(a, b) { var c = hr[b] || [];
            c.push(a);
            hr[b] = c },
        lr = function() {
            T.addTargetToGroup = function(e) { kr(e, "default") };
            T.addDestinationToContainer = function(e, f) { jr(e, f) };
            var a = T.pendingDefaultTargets;
            delete T.pendingDefaultTargets;
            if (Array.isArray(a))
                for (var b = 0; b < a.length; ++b) kr(a[b], "default");
            var c = T.pendingDestinationIds;
            delete T.pendingDestinationIds;
            if (Array.isArray(c))
                for (var d = 0; d < c.length; ++d) jr(c[d][0], c[d][1])
        },
        kr = function(a, b) { b = b.toString().split(","); for (var c = 0; c < b.length; c++) { var d = gr[b[c]] || [];
                gr[b[c]] = d;
                0 > d.indexOf(a) && d.push(a) } },
        mr = function(a) { Ma(gr, function(b, c) { var d = c.indexOf(a);
                0 <= d && c.splice(d, 1) }) };
    var Qr = function(a) { var b = T.zones; return b ? b.getIsAllowedFn(J.J, a) : function() { return !0 } },
        Rr = function(a) { var b = T.zones; return b ? b.isActive(J.J, a) : !0 };
    var Sr = function(a) { return arguments },
        Vr = function(a, b) {
            for (var c = [], d = 0; d < le.length; d++)
                if (a[d]) { var e = le[d]; var f = ul(b.Hb); try { var g = Jm(d, { onSuccess: f, onFailure: f, terminate: f }, b, d); if (g) { var h = c,
                                k = h.push,
                                n = d,
                                p = e["function"]; if (!p) throw "Error: No function name given for function call."; var q = ne[p];
                            k.call(h, { Dh: n, uh: q ? q.priorityOverride || 0 : 0, execute: g }) } else Tr(d, b), f() } catch (t) { f() } }
            var r = b.Hb;
            r.P = !0;
            r.o >= r.s &&
                tl(r);
            c.sort(Ur);
            for (var u = 0; u < c.length; u++) c[u].execute();
            return 0 < c.length
        };

    function Ur(a, b) { var c, d = b.uh,
            e = a.uh;
        c = d > e ? 1 : d < e ? -1 : 0; var f; if (0 !== c) f = c;
        else { var g = a.Dh,
                h = b.Dh;
            f = g > h ? 1 : g < h ? -1 : 0 } return f }

    function Tr(a, b) { if (!um) return; var c = function(d) { var e = b.zf(le[d]) ? "3" : "4",
                f = te(le[d][Ld.Hg], b, []);
            f && f.length && c(f[0].index);
            Am(b.id, le[d], e); var g = te(le[d][Ld.Kg], b, []);
            g && g.length && c(g[0].index) };
        c(a); }
    var Wr = !1,
        Xr;
    var cs = function(a) {
        var b = Ua(),
            c = a["gtm.uniqueEventId"],
            d = a.event;
        if ("gtm.js" === d) { if (Wr) return !1;
            Wr = !0; }
        var g, h = !1;
        if (Rr(c)) g = Qr(c);
        else { if ("gtm.js" !== d && "gtm.init" !== d && "gtm.init_consent" !== d) return !1;
            h = !0;
            g = Qr(Number.MAX_SAFE_INTEGER) }
        zm(c, d);
        var k = a.eventCallback,
            n = a.eventTimeout,
            p = k;
        var q = { id: c, name: d, zf: hl(g), nk: [], ph: function() { jg(6) }, Ug: Zr(), Vg: $r(c), Hb: new ql(p, n) },
            r = De(q);
        h && (r =
            as(r));
        var u = Vr(r, q);
        "gtm.js" !==
        d && "gtm.sync" !== d || Fl(J.J);
        return bs(r, u)
    };

    function $r(a) { return function(b) { um && (tc(b) || Im(a, "input", b)) } }

    function Zr() { var a = {};
        a.event = Rh("event", 1);
        a.ecommerce = Rh("ecommerce", 1);
        a.gtm = Rh("gtm");
        a.eventModel = Rh("eventModel"); return a }

    function as(a) { for (var b = [], c = 0; c < a.length; c++) a[c] && (vh[String(le[c][Ld.Eb])] && (b[c] = !0), void 0 !== le[c][Ld.Mi] && (b[c] = !0)); return b }

    function bs(a, b) { if (!b) return b; for (var c = 0; c < a.length; c++)
            if (a[c] && le[c] && !wh[String(le[c][Ld.Eb])]) return !0;
        return !1 }
    var Le;
    var ds = "HA GF G UA AW DC".split(" "),
        es = !1,
        fs = !1,
        gs = 0;

    function hs(a) { a.hasOwnProperty("gtm.uniqueEventId") || Object.defineProperty(a, "gtm.uniqueEventId", { value: Gh() }); return a["gtm.uniqueEventId"] }

    function is() { es || T.gtagRegistered || (es = T.gtagRegistered = !0, lr()); return es }
    var js = {
            config: function(a) { hs(a); if (2 > a.length || !Da(a[1])) return; var b = {}; if (2 < a.length) { if (void 0 != a[2] && !oc(a[2]) || 3 < a.length) return;
                    b = a[2] } var c = oo(a[1]); if (!c) return; var d = is();
                mr(c.id);
                kr(c.id, b[S.Ne] || "default");
                delete b[S.Ne];
                fs || jg(43); if (d && -1 !== ds.indexOf(c.prefix)) { "G" === c.prefix && (b[S.Db] = !0);
                    delete b[S.bc];
                    c.id === c.containerId && (T.configCount = ++gs);
                    Vq(b, c.id); return }
                Ph("gtag.targets." + c.id, void 0);
                Ph("gtag.targets." + c.id, pc(b)); },
            consent: function(a, b) { if (3 === a.length) { jg(39); var c = Gh(),
                        d = a[1]; "default" === d ? Hg(a[2]) : "update" === d && Jg(a[2], c, b) } },
            event: function(a) {
                var b = a[1];
                if (!(2 > a.length) && Da(b)) {
                    var c;
                    if (2 < a.length) { if (!oc(a[2]) && void 0 != a[2] || 3 < a.length) return;
                        c = a[2] }
                    var d = c,
                        e = {},
                        f = (e.event = b, e);
                    d && (f.eventModel = pc(d), d[S.bc] && (f.eventCallback = d[S.bc]), d[S.Dd] && (f.eventTimeout = d[S.Dd]));
                    var g = hs(a);
                    f["gtm.uniqueEventId"] = g;
                    if ("optimize.callback" === b) return f.eventModel = f.eventModel || {}, f;
                    var h;
                    var k = c && c[S.hc];
                    void 0 === k && (k = Mh(S.hc, 2), void 0 === k && (k = "default"));
                    if (Da(k) || Ha(k)) { for (var n = k.toString().replace(/\s+/g, "").split(","), p = [], q = 0; q < n.length; q++)
                            if (0 <= n[q].indexOf("-")) er && fr ? -1 < (hr[J.J] || []).indexOf(n[q]) && p.push(n[q]) : p.push(n[q]);
                            else { var r = gr[n[q]] || []; if (er)
                                    if (fr) { if (-1 < r.indexOf(J.J)) { var u = hr[J.J];
                                            u && u.length && (p = p.concat(u)) } } else
                                        for (var t = 0; t < r.length; t++) { var v = hr[r[t]];
                                            v && v.length && (p = p.concat(v)) } else r && r.length && (p = p.concat(r)) }
                        h = qo(p) } else h = void 0;
                    var x = h;
                    if (!x) return;
                    for (var y = is(), w = [], A = 0; y && A < x.length; A++) { var B = x[A]; if (-1 !== ds.indexOf(B.prefix)) { var C = pc(c); "G" === B.prefix && (C[S.Db] = !0);
                            delete C[S.bc];
                            Tq(b, C, B.id) }
                        w.push(B.id) }
                    f.eventModel = f.eventModel || {};
                    0 < x.length ? f.eventModel[S.hc] = w.join() : delete f.eventModel[S.hc];
                    fs || jg(43);
                    return f
                }
            },
            get: function(a) {
                jg(53);
                if (4 !== a.length || !Da(a[1]) || !Da(a[2]) || !Ba(a[3])) return;
                var b = oo(a[1]),
                    c = String(a[2]),
                    d = a[3];
                if (!b) return;
                fs || jg(43);
                if (!is() || -1 === ds.indexOf(b.prefix)) return;
                Gh();
                var e = {};
                Dg(pc((e[S.eb] = c, e[S.pb] = d, e)));
                Wq(c, function(f) { I(function() { return d(f) }) }, b.id);
            },
            js: function(a) { if (2 == a.length && a[1].getTime) { fs = !0;
                    is(); var b = {}; return b.event = "gtm.js", b["gtm.start"] = a[1].getTime(), b["gtm.uniqueEventId"] = hs(a), b } },
            policy: function(a) { if (3 === a.length) { var b = a[1],
                        c = a[2],
                        d = Le.o;
                    d.g[b] ? d.g[b].push(c) : d.g[b] = [c] } },
            set: function(a) {
                var b;
                2 == a.length && oc(a[1]) ? b = pc(a[1]) : 3 == a.length && Da(a[1]) && (b = {}, oc(a[2]) || Ha(a[2]) ? b[a[1]] = pc(a[2]) : b[a[1]] =
                    a[2]);
                if (b) { if (Gh(), pc(b), is()) { var c = pc(b);
                        Qq().push("set", [c]) }
                    b._clear = !0; return b }
            }
        },
        ks = { policy: !0 };
    var ls = function() { this.g = [];
        this.o = [] };
    ls.prototype.push = function(a, b, c) { var d = this.g.length + 1;
        c = pc(c);
        c.priorityId = d; var e = { debugContext: c, message: a, notBeforeEventId: b, priorityId: d };
        this.g.push(e); for (var f = 0; f < this.o.length; f++) try { this.o[f](e) } catch (g) {} };
    ls.prototype.listen = function(a) { this.o.push(a) };
    ls.prototype.get = function() { for (var a = {}, b = 0; b < this.g.length; b++) { var c = this.g[b],
                d = a[c.notBeforeEventId];
            d || (d = [], a[c.notBeforeEventId] = d);
            d.push(c) } return a };

    function ms(a, b) { return a.notBeforeEventId - b.notBeforeEventId || a.priorityId - b.priorityId };
    var ns = function(a) { var b = m[J.Z].hide; if (b && void 0 !== b[a] && b.end) { b[a] = !1; var c = !0,
                    d; for (d in b)
                    if (b.hasOwnProperty(d) && !0 === b[d]) { c = !1; break }
                c && (b.end(), b.end = null) } },
        os = function(a) { var b = m[J.Z],
                c = b && b.hide;
            c && c.end && (c[a] = !0) };
    var ps = !1,
        qs = [];

    function rs() { if (!ps) { ps = !0; for (var a = 0; a < qs.length; a++) I(qs[a]) } }
    var ss = function(a) { ps ? I(a) : qs.push(a) };
    var Js = function(a) { if (Is(a)) return a;
        this.g = a };
    Js.prototype.vj = function() { return this.g };
    var Is = function(a) { return !a || "object" !== mc(a) || oc(a) ? !1 : "getUntrustedMessageValue" in a };
    Js.prototype.getUntrustedMessageValue = Js.prototype.vj;
    var Ks = 0,
        Ls, Ms = {},
        Ns = [],
        Os = [],
        Ps = !1,
        Qs = !1,
        Rs = function(a) { return m[J.Z].push(a) },
        Ss = function(a, b, c) { a.eventCallback = b;
            c && (a.eventTimeout = c); return Rs(a) },
        Ts = function(a, b) { var c = T[J.Z],
                d = c ? c.subscribers : 1,
                e = 0,
                f = !1,
                g = void 0;
            b && (g = m.setTimeout(function() { f || (f = !0, a());
                g = void 0 }, b)); return function() {++e === d && (g && (m.clearTimeout(g), g = void 0), f || (a(), f = !0)) } };

    function Us(a) { var b = a._clear;
        Ma(a, function(d, e) { "_clear" !== d && (b && Ph(d, void 0), Ph(d, e)) });
        Bh || (Bh = a["gtm.start"]); var c = a["gtm.uniqueEventId"]; if (!a.event) return !1;
        c || (c = Gh(), a["gtm.uniqueEventId"] = c, Ph("gtm.uniqueEventId", c)); return cs(a) }

    function Vs(a) { if (null == a || "object" !== typeof a) return !1; if (a.event) return !0; if (Na(a)) { var b = a[0]; if ("config" === b || "event" === b || "js" === b) return !0 } return !1 }

    function Ws() {
        for (var a = !1; !Qs && (0 < Ns.length || 0 < Os.length);) {
            if (!Ps && Vs(Ns[0])) { var b = {},
                    c = (b.event = "gtm.init_consent", b),
                    d = {},
                    e = (d.event = "gtm.init", d),
                    f = Ns[0]["gtm.uniqueEventId"];
                f && (c["gtm.uniqueEventId"] = f - 2, e["gtm.uniqueEventId"] = f - 1);
                Ns.unshift(c, e);
                Ps = !0 }
            Qs = !0;
            delete Jh.eventModel;
            Lh();
            var g = null,
                h = void 0;
            null == g && (g = Ns.shift());
            if (null != g) {
                var n = Is(g);
                if (n) { var p = g;
                    g = Is(p) ? p.getUntrustedMessageValue() : void 0;
                    Qh() }
                try {
                    if (Ba(g)) try { g.call(Nh) } catch (B) {} else if (Ha(g)) { var q = g; if (Da(q[0])) { var r = q[0].split("."),
                                u = r.pop(),
                                t = q.slice(1),
                                v = Mh(r.join("."), 2); if (null != v) try { v[u].apply(v, t) } catch (B) {} } } else {
                        if (Na(g)) {
                            a: {
                                var x = g,
                                    y = h;
                                if (x.length && Da(x[0])) { var w = js[x[0]]; if (w && (!n || !ks[x[0]])) { g = w(x, y); break a } }
                                g =
                                void 0
                            }
                            if (!g) { Qs = !1; continue }
                        }
                        a = Us(g) || a;
                    }
                } finally { n && Lh(!0) }
            }
            Qs = !1
        }
        return !a
    }

    function Ys() { var b = Ws(); try { ns(J.J) } catch (c) {} return b }
    var at = function() { var a = Jb(J.Z, []),
            b = Jb("google_tag_manager", {});
        b = b[J.Z] = b[J.Z] || {};
        nl(function() { if (!b.gtmDom) { b.gtmDom = !0; var e = {};
                a.push((e.event = "gtm.dom", e)) } });
        ss(function() { if (!b.gtmLoad) { b.gtmLoad = !0; var e = {};
                a.push((e.event = "gtm.load", e)) } });
        b.subscribers = (b.subscribers || 0) + 1; var c = a.push;
        a.push = function() { var e; if (0 < T.SANDBOXED_JS_SEMAPHORE) { e = []; for (var f = 0; f < arguments.length; f++) e[f] = new Js(arguments[f]) } else e = [].slice.call(arguments, 0);
            Ns.push.apply(Ns, e); var g = c.apply(a, e); if (300 < this.length)
                for (jg(4); 300 < this.length;) this.shift(); var h = "boolean" !== typeof g || g; return Ws() && h }; var d = a.slice(0);
        Ns.push.apply(Ns, d); if (Zs()) { I(Ys) } };
    var Zs = function() { var a = !0; return a };

    function bt(a) { if (null == a || 0 === a.length) return !1; var b = Number(a),
            c = Ua(); return b < c + 3E5 && b > c - 9E5 };
    var ct = { sh: "G-CD35DZJ728" },
        dt = function() { var a = [];
            ct.sh && (a = ct.sh.split("|")); return a };
    var et = {};
    et.Ld = new String("undefined");
    var ht = function(a, b, c) { var d = { event: b, "gtm.element": a, "gtm.elementClasses": Zb(a, "className"), "gtm.elementId": a["for"] || Ub(a, "id") || "", "gtm.elementTarget": a.formTarget || Zb(a, "target") || "" };
            c && (d["gtm.triggers"] = c.join(","));
            d["gtm.elementUrl"] = (a.attributes && a.attributes.formaction ? a.formAction : "") || a.action || Zb(a, "href") || a.src || a.code || a.codebase || ""; return d },
        it = function(a) {
            T.hasOwnProperty("autoEventsSettings") || (T.autoEventsSettings = {});
            var b = T.autoEventsSettings;
            b.hasOwnProperty(a) || (b[a] = {});
            return b[a]
        },
        jt = function(a, b, c) { it(a)[b] = c },
        kt = function(a, b, c, d) { var e = it(a),
                f = Va(e, b, d);
            e[b] = c(f) },
        lt = function(a, b, c) { var d = it(a); return Va(d, b, c) },
        mt = function(a) { return "string" === typeof a ? a : String(Gh()) };
    var st = !!m.MutationObserver,
        tt = void 0,
        ut = function(a) { if (!tt) { var b = function() { var c = H.body; if (c)
                        if (st)(new MutationObserver(function() { for (var e = 0; e < tt.length; e++) I(tt[e]) })).observe(c, { childList: !0, subtree: !0 });
                        else { var d = !1;
                            Sb(c, "DOMNodeInserted", function() { d || (d = !0, I(function() { d = !1; for (var e = 0; e < tt.length; e++) I(tt[e]) })) }) } };
                tt = [];
                H.body ? b() : I(b) }
            tt.push(a) };
    var Ft = function(a, b, c) {
        function d() { var g = a();
            f += e ? (Ua() - e) * g.playbackRate / 1E3 : 0;
            e = Ua() }
        var e = 0,
            f = 0;
        return {
            createEvent: function(g, h, k) {
                var n = a(),
                    p = n.sf,
                    q = void 0 !== k ? Math.round(k) : void 0 !== h ? Math.round(n.sf * h) : Math.round(n.$g),
                    r = void 0 !== h ? Math.round(100 * h) : 0 >= p ? 0 : Math.round(q / p * 100),
                    u = H.hidden ? !1 : .5 <= Yh(c);
                d();
                var t = void 0;
                void 0 !== b && (t = [b]);
                var v = ht(c, "gtm.video", t);
                v["gtm.videoProvider"] = "youtube";
                v["gtm.videoStatus"] = g;
                v["gtm.videoUrl"] = n.url;
                v["gtm.videoTitle"] = n.title;
                v["gtm.videoDuration"] =
                    Math.round(p);
                v["gtm.videoCurrentTime"] = Math.round(q);
                v["gtm.videoElapsedTime"] = Math.round(f);
                v["gtm.videoPercent"] = r;
                v["gtm.videoVisible"] = u;
                return v
            },
            Ah: function() { e = Ua() },
            nc: function() { d() }
        }
    };
    var Gt = m.clearTimeout,
        Ht = m.setTimeout,
        U = function(a, b, c, d) { if (Tm()) { b && I(b) } else return Ob(a, b, c, d) },
        It = function() { return new Date },
        Jt = function() { return m.location.href },
        Kt = function(a) { return ii(ki(a), "fragment") },
        Lt = function(a) { return ji(ki(a)) },
        Mt = function(a, b) { return Mh(a, b || 2) },
        Nt = function(a, b, c) { return b ? Ss(a, b, c) : Rs(a) },
        Ot = function(a, b) { m[a] = b },
        V = function(a, b, c) { b && (void 0 === m[a] || c && !m[a]) && (m[a] = b); return m[a] },
        Pt = function(a, b, c) { return Mi(a, b, void 0 === c ? !0 : !!c) },
        Qt = function(a, b, c) { return 0 === Vi(a, b, c) },
        Rt = function(a, b) { if (Tm()) { b && I(b) } else Qb(a, b) },
        St = function(a) { return !!lt(a, "init", !1) },
        Tt = function(a) { jt(a, "init", !0) },
        Ut = function(a, b, c) { um && (tc(a) || Im(c, b, a)) };
    var ru = ["matches", "webkitMatchesSelector", "mozMatchesSelector", "msMatchesSelector", "oMatchesSelector"];

    function su(a, b) { a = String(a);
        b = String(b); var c = a.length - b.length; return 0 <= c && a.indexOf(b, c) === c }
    var tu = new Ka;

    function uu(a, b, c) { var d = c ? "i" : void 0; try { var e = String(b) + d,
                f = tu.get(e);
            f || (f = new RegExp(b, d), tu.set(e, f)); return f.test(a) } catch (g) { return !1 } }

    function vu(a, b) {
        function c(g) { var h = ki(g),
                k = ii(h, "protocol"),
                n = ii(h, "host", !0),
                p = ii(h, "port"),
                q = ii(h, "path").toLowerCase().replace(/\/$/, ""); if (void 0 === k || "http" === k && "80" === p || "https" === k && "443" === p) k = "web", p = "default"; return [k, n, p, q] } for (var d = c(String(a)), e = c(String(b)), f = 0; f < d.length; f++)
            if (d[f] !== e[f]) return !1;
        return !0 }

    function wu(a) { return xu(a) ? 1 : 0 }

    function xu(a) {
        var b = a.arg0,
            c = a.arg1;
        if (a.any_of && Array.isArray(c)) { for (var d = 0; d < c.length; d++) { var e = pc(a, {});
                pc({ arg1: c[d], any_of: void 0 }, e); if (wu(e)) return !0 } return !1 }
        switch (a["function"]) {
            case "_cn":
                return 0 <= String(b).indexOf(String(c));
            case "_css":
                var f;
                a: { if (b) try { for (var g = 0; g < ru.length; g++) { var h = ru[g]; if (b[h]) { f = b[h](c); break a } } } catch (k) {}
                    f = !1 }
                return f;
            case "_ew":
                return su(b, c);
            case "_eq":
                return String(b) === String(c);
            case "_ge":
                return Number(b) >= Number(c);
            case "_gt":
                return Number(b) > Number(c);
            case "_lc":
                return 0 <= String(b).split(",").indexOf(String(c));
            case "_le":
                return Number(b) <= Number(c);
            case "_lt":
                return Number(b) < Number(c);
            case "_re":
                return uu(b, c, a.ignore_case);
            case "_sw":
                return 0 === String(b).indexOf(String(c));
            case "_um":
                return vu(b, c)
        }
        return !1
    };

    function yu(a, b) { var c = this; };
    var zu = !0;

    function Au(a, b, c) { var d; return d }

    function Bu(a, b, c) {};

    function Cu(a, b, c, d) {};

    function Du(a) {};

    function Hu(a) {};
    var Iu = {},
        Ju = [],
        Ku = {},
        Lu = 0,
        Mu = 0;

    function Tu(a, b) {}

    function $u(a, b) {};

    function ev(a) {};
    var fv = {},
        gv = [];
    var nv = function(a, b) {};

    function ov(a, b, c) {};

    function pv(a, b) { return !0 };

    function qv(a, b, c) {};

    function rv(a, b) { var c; return c };

    function sv(a) {};

    function tv(a) {};

    function uv(a) { M(G(this), ["listener:!Fn"], arguments);
        N(this, "process_dom_events", "window", "load");
        ss(rc(a)) };

    function vv(a) { var b; return b };

    function wv(a, b) { var c; var d = !1; var e = qc(c, this.g, d);
        void 0 === e && void 0 !== c && jg(45); return e };

    function xv(a) { var b; return b };

    function yv(a, b) { var c; return c };

    function zv(a, b) { var c = null,
            d = !1; return qc(c, this.g, d) };

    function Av(a) { var b; var g = !1; return qc(b, this.g, g) };
    var Bv = {},
        Cv = [],
        Dv = {},
        Ev = 0,
        Fv = 0;
    var Lv = !0;
    var Mv = function(a, b) { return b };
    var Rv = !0;

    function Sv(a, b) { return b }
    var Xv = !0;

    function Yv() {}
    var Zv = {},
        $v = [];
    var gw = !0;

    function hw(a, b) { return b }
    var lw = !0;

    function mw(a, b) { return b }
    var nw, ow;
    var xw = !0;
    var yw = function(a, b) { return b };
    var Fb = ca(["data-gtm-yt-inspected-"]),
        zw = ["www.youtube.com", "www.youtube-nocookie.com"],
        Aw, Bw = !1;
    var Lw = !0;

    function Mw(a, b) { return b }

    function Nw(a) { return !1 }
    var Ow;

    function Pw(a) { var b = !1; return b };
    var Rw = function(a, b, c) { if (c) switch (c.type) {
                case "event_name":
                    return a;
                case "const":
                    return c.const_value;
                case "event_param":
                    var d = c.event_param.param_name; return b[d] } },
        Uw = function(a, b, c, d) { if (c && !Sw(a, b, c)) return !1; if (!d || 0 === d.length) return !0; for (var e = 0; e < d.length; e++)
                if (Tw(a, b, d[e].predicates || [])) return !0;
            return !1 },
        Tw = function(a, b, c) {
            for (var d = 0; d < c.length; d++)
                if (!Sw(a,
                        b, c[d])) return !1;
            return !0
        },
        Sw = function(a, b, c) {
            var d = c.values || [],
                e = Rw(a, b, d[0]),
                f = Rw(a, b, d[1]),
                g = c.type;
            if ("eqi" === g || "swi" === g || "ewi" === g || "cni" === g) Da(e) && (e = e.toLowerCase()), Da(f) && (f = f.toLowerCase());
            var h = !1;
            switch (g) {
                case "eq":
                case "eqi":
                    h = String(e) === String(f);
                    break;
                case "sw":
                case "swi":
                    h = 0 === String(e).indexOf(String(f));
                    break;
                case "ew":
                case "ewi":
                    h = su(e, f);
                    break;
                case "cn":
                case "cni":
                    h = 0 <= String(e).indexOf(String(f));
                    break;
                case "lt":
                    h = Number(e) < Number(f);
                    break;
                case "le":
                    h = Number(e) <= Number(f);
                    break;
                case "gt":
                    h = Number(e) > Number(f);
                    break;
                case "ge":
                    h = Number(e) >= Number(f);
                    break;
                case "re":
                case "rei":
                    h = uu(e, f, "rei" === g)
            }
            return !!c.negate !== h
        };

    function Vw(a, b) { var c = !1; return c };
    var Ww = function(a) { var b; return b };

    function Xw(a, b) { b = void 0 === b ? !0 : b; var c; return c };

    function Yw() { return eg.Yg };

    function Zw() { var a = []; return qc(a) };

    function $w(a) { var b = null; return b };

    function ax(a, b) { var c; return c };

    function bx(a, b) { var c; return c };

    function cx(a, b) { var c; return c };

    function dx(a) { var b = ""; return b };

    function ex() { return eg.xh };

    function fx(a, b) { var c; return c };

    function gx(a) { var b = ""; return b };

    function hx() { N(this, "get_user_agent"); return m.navigator.userAgent };

    function ix(a) { return a ? { entityType: a.dh.type, indexInOriginContainer: a.dh.index, nameInOriginContainer: void 0, originContainerId: J.J } : {} };

    function kx(a, b) {};

    function lx(a, b) {};
    var mx = {};

    function nx(a, b, c, d, e, f) { f ? e[f] ? (e[f][0].push(c), e[f][1].push(d)) : (e[f] = [
            [c],
            [d]
        ], Ob(a, function() { for (var g = e[f][0], h = 0; h < g.length; h++) I(g[h]);
            g.push = function(k) { I(k); return 0 } }, function() { for (var g = e[f][1], h = 0; h < g.length; h++) I(g[h]);
            e[f] = null }, b)) : Ob(a, c, d, b) }

    function ox(a, b, c, d) {}
    var px = Object.freeze({ dl: 1, id: 1 }),
        qx = {};

    function rx(a, b, c, d) {};

    function sx(a) { var b = !0; return b };
    var tx = function() { return !1 },
        ux = {
            getItem: function(a) { var b = null; return b },
            setItem: function(a,
                b) { return !1 },
            removeItem: function(a) {}
        };
    var vx = ["textContent", "value", "tagName", "children", "childElementCount"];

    function wx(a) { var b; return b };

    function xx() {};

    function yx(a, b) { var c; return c };

    function zx(a) { var b = void 0; return b };

    function Ax(a) {};

    function Bx(a, b) { var c = !1; return c };

    function Cx() { var a = ""; return a };

    function Dx() { var a = ""; return a };

    function Ex(a, b) { var c = this; };
    var Fx = Object.freeze(["config", "event", "get", "set"]);

    function Gx(a, b, c) {};
    var Hx = !0;

    function Ix(a, b) { var c = !1; return c }

    function Jx() {};

    function Kx(a, b, c, d) { d = void 0 === d ? !1 : d; };

    function Lx(a, b, c) {};

    function Mx(a, b, c, d) { var e = this;
        d = void 0 === d ? !0 : d; var f = !1; return f };
    var Nx = !1;

    function Ox(a) { M(G(this), ["consentSettings:!DustMap"], arguments); for (var b = a.Fb(), c = b.length(), d = 0; d < c; d++) { var e = b.get(d);
            e !== S.te && N(this, "access_consent", e, "write") } var f = this.g.g,
            g = ix(f); if (Nx) { var h = Sr("consent", "default", rc(a)),
                k = f.eventId;
            Ls.push(new Js(h), k, g) } else Hg(rc(a)) }

    function Px(a, b, c) { return !1 };

    function Qx(a, b, c) { M(G(this), ["targetId:!string", "name:!string", "value:!*"], arguments); var d = Vh(a) || {};
        d[b] = rc(c, this.g); var e = a;
        Th || Uh();
        Sh[e] = d; };

    function Rx(a, b, c) {};
    var Sx = function(a) { for (var b = [], c = 0, d = 0; d < a.length; d++) { var e = a.charCodeAt(d);
            128 > e ? b[c++] = e : (2048 > e ? b[c++] = e >> 6 | 192 : (55296 == (e & 64512) && d + 1 < a.length && 56320 == (a.charCodeAt(d + 1) & 64512) ? (e = 65536 + ((e & 1023) << 10) + (a.charCodeAt(++d) & 1023), b[c++] = e >> 18 | 240, b[c++] = e >> 12 & 63 | 128) : b[c++] = e >> 12 | 224, b[c++] = e >> 6 & 63 | 128), b[c++] = e & 63 | 128) } return b };

    function Tx(a, b, c, d) { var e = this; };

    function Ux(a, b, c) {};
    var Vx = {},
        Wx = {};
    Vx.getItem = function(a) { var b = null; return b };
    Vx.setItem = function(a, b) {};
    Vx.removeItem = function(a) {};
    Vx.clear = function() {};
    var Xx = function(a) { var b; return b };
    var Yx = !1;

    function Zx(a) { M(G(this), ["consentSettings:!DustMap"], arguments); var b = rc(a),
            c; for (c in b) b.hasOwnProperty(c) && N(this, "access_consent", c, "write"); var d = this.g.g,
            e = ix(d); if (Yx) { var f = d.eventId;
            Ls.push(new Js(Sr("consent", "update", b)), f, e) } else Jg(b, d.eventId, e) }
    var $x = function() {
        var a = new Wf;
        Tm() ? (a.add("injectHiddenIframe", Aa), a.add("injectScript", Aa)) : (a.add("injectHiddenIframe", lx), a.add("injectScript", ox));
        var b = Lx;
        a.add("Math", Af());
        a.add("Object", Uf);
        a.add("TestHelper", Zf());
        a.add("addConsentListener", yu);
        a.add("addEventCallback", Du);
        a.add("aliasInWindow", pv);
        a.add("assertApi",
            wf);
        a.add("assertThat", xf);
        a.add("callInWindow", rv);
        a.add("callLater", sv);
        a.add("copyFromDataLayer", wv);
        a.add("copyFromWindow", xv);
        a.add("createArgumentsQueue", zv);
        a.add("createQueue", Av);
        a.add("decodeUri", Bf);
        a.add("decodeUriComponent", Cf);
        a.add("encodeUri", Df);
        a.add("encodeUriComponent", Ef);
        a.add("fail", Ff);
        a.add("fromBase64", Ww, !("atob" in m));
        a.add("generateRandom", Gf);
        a.add("getContainerVersion", Hf);
        a.add("getCookieValues", Xw);
        a.add("getQueryParameters", bx);
        a.add("getReferrerQueryParameters",
            cx);
        a.add("getReferrerUrl", dx);
        a.add("getTimestamp", Kf);
        a.add("getTimestampMillis", Kf);
        a.add("getType", Lf);
        a.add("getUrl", gx);
        a.add("isConsentGranted", sx);
        a.add("localStorage", ux, !tx());
        a.add("logToConsole", xx);
        a.add("makeInteger", Nf);
        a.add("makeNumber", Of);
        a.add("makeString", Pf);
        a.add("makeTableMap", Qf);
        a.add("mock", Tf);
        a.add("parseUrl", zx);
        a.add("queryPermission", Bx);
        a.add("readCharacterSet", Cx);
        a.add("readTitle", Dx);
        a.add("sendPixel", b);
        a.add("setCookie", Mx);
        a.add("setDefaultConsentState", Ox);
        a.add("setInWindow", Px);
        a.add("sha256", Tx);
        a.add("templateStorage", Vx);
        a.add("toBase64", Xx, !("btoa" in m));
        a.add("updateConsentState", Zx);
        Yf(a, "callOnWindowLoad", uv);
        Yf(a, "internal.addFormInteractionListener", Tu);
        Yf(a, "internal.addFormSubmitListener", $u);
        Yf(a, "internal.addGaSendListener", ev);
        Yf(a, "internal.addHistoryChangeListener", nv);
        Yf(a, "internal.evaluateFilteringRules", Nw);
        Yf(a, "internal.evaluateMatchingRules", Pw);
        Yf(a, "internal.getFlags", Jf);
        Yf(a, "internal.locateUserData", wx);
        Yf(a, "internal.registerGtagCommandListener",
            Gx);
        Yf(a, "internal.sendGtagEvent", Kx);
        a.add("JSON", Mf(function(c) { var d = this.g.g;
            d && d.log.call(this, "error", c) }));
        Yf(a, "internal.appendRemoteConfigParameter", qv), Yf(a, "internal.getRemoteConfigParameter", fx), Yf(a, "internal.setRemoteConfigParameter", Rx), Yf(a, "internal.sortRemoteConfigParameters", Ux), Yf(a, "internal.getProductSettingsParameter", ax), Yf(a, "internal.setProductSettingsParameter",
            Qx);
        Yf(a, "internal.getDestinationIds", Zw);
        Tm() ? Yf(a, "internal.injectScript", Aa) : Yf(a, "internal.injectScript", rx);

        return function(c) {
            var d;
            if (a.g.hasOwnProperty(c)) d = a.get(c, this);
            else {
                var e;
                if (e = a.o.hasOwnProperty(c)) { var f = !1,
                        g = this.g.g; if (g) { var h = g.dd(); if (h) { 0 !== h.indexOf("__cvt_") && (f = !0); } }
                    e = f }
                if (e) {
                    var k = a.o.hasOwnProperty(c) ? a.o[c] : void 0;
                    d = k
                } else throw Error(c + " is not a valid API name.");
            }
            return d
        }
    };
    var ay = function() { return !1 },
        by = function() { var a = {}; return function(b, c, d) {} };
    var cy;

    function dy() {
        var a = cy;
        return function(b, c, d) {
            var e = d && d.event;
            ey(c);
            var f = new ib;
            Ma(c, function(q, r) { var u = qc(r);
                void 0 === u && void 0 !== r && jg(44);
                f.set(q, u) });
            a.g.g.I = Ae();
            var g = { Wi: Me(b), eventId: void 0 !== e ? e.id : void 0, Sd: void 0 !== e ? function(q) { return e.Hb.Sd(q) } : void 0, dd: function() { return b }, log: function() {}, dh: { index: d && d.index, type: d && d.type } };
            if (ay()) {
                var h =
                    by(),
                    k = void 0,
                    n = void 0;
                g.Oa = { Pf: [], Xc: {}, Va: function(q, r, u) { 1 === r && (k = q);
                        7 === r && (n = u);
                        h(q, r, u) }, Ef: Rf() };
                g.log = function(q, r) { if (k) { var u = Array.prototype.slice.call(arguments, 1);
                        h(k, 4, { level: q, source: n, message: u }) } }
            }
            var p = Kd(a, g, [b, f]);
            a.g.g.I = void 0;
            p instanceof oa && "return" === p.g && (p = p.o);
            return rc(p)
        }
    }

    function ey(a) { var b = a.gtmOnSuccess,
            c = a.gtmOnFailure;
        Ba(b) && (a.gtmOnSuccess = function() { I(b) });
        Ba(c) && (a.gtmOnFailure = function() { I(c) }) }

    function fy() { cy.g.g.P = function(a, b, c) { T.SANDBOXED_JS_SEMAPHORE = T.SANDBOXED_JS_SEMAPHORE || 0;
            T.SANDBOXED_JS_SEMAPHORE++; try { return a.apply(b, c) } finally { T.SANDBOXED_JS_SEMAPHORE-- } } }

    function gy(a) { void 0 !== a && Ma(a, function(b, c) { for (var d = 0; d < c.length; d++) { var e = c[d].replace(/^_*/, "");
                Fh[e] = Fh[e] || [];
                Fh[e].push(b) } }) };
    var hy = encodeURI,
        Y = encodeURIComponent,
        iy = Rb;
    var jy = function(a, b) { if (!a) return !1; var c = ii(ki(a), "host"); if (!c) return !1; for (var d = 0; b && d < b.length; d++) { var e = b[d] && b[d].toLowerCase(); if (e) { var f = c.length - e.length;
                0 < f && "." != e.charAt(0) && (f--, e = "." + e); if (0 <= f && c.indexOf(e, f) == f) return !0 } } return !1 };
    var ky = function(a, b, c) { for (var d = {}, e = !1, f = 0; a && f < a.length; f++) a[f] && a[f].hasOwnProperty(b) && a[f].hasOwnProperty(c) && (d[a[f][b]] = a[f][c], e = !0); return e ? d : null };
    var uz = !1;
    var vz = {};
    vz[S.$b] = "";
    vz[S.Ia] = "";
    vz[S.Ra] = "";
    vz[S.na] = "auto";
    vz[S.nb] = "/";
    vz[S.za] = 63072E3;
    vz[S.Qc] = 30;
    vz[S.Hd] = 1E4;
    vz[S.Aa] = !0;
    var wz = function(a, b, c, d, e, f) {
        f = void 0 === f ? !1 : f;
        up.call(this, a, b, d);
        this.o = c;
        this.M = d.eventModel;
        this.Ag = d.containerConfig;
        uz || (this.I = d.getWithConfig(S.Ha) ? 1 : 8, this.Ig = this.jc = !1);
        uz || (this.Xe = d.getWithConfig(S.ob) ? 1 : 8);
        this.C =
            e;
        this.P = this.We = this.Pg = this.s = this.Lg = !1;
        this.Nb = 0;
        this.af = this.xb = !1;
        this.bf = void 0;
        this.ab = 0;
        this.Og = this.g = this.Ze = this.Eg = void 0;
        this.Mg = !1;
        this.Cg = this.$e = void 0;
        this.Pi = f
    };
    la(wz, up);
    l = wz.prototype;
    l.Ta = function(a, b) { void 0 === this.B(a) && (this.M[a] = b) };
    l.B = function(a) { return void 0 !== this.M[a] ? this.M[a] : void 0 !== this.H.getWithConfig(a) ? this.H.getWithConfig(a) : void 0 !== this.C[a] ? this.C[a] : vz[a] };
    l.getRemoteConfig = function(a) { return void 0 !== this.H.remoteConfig[a] ? this.H.remoteConfig[a] : this.C[a] };
    l.Lb = function(a) { var b = wp(this, S.Cd, this.C[S.Cd]); if (b && void 0 !== b[a || this.eventName]) return b[a || this.eventName] };
    l.ih = function() { var a;
        a = uz ? this.metadata.user_data : this.B(S.Ca); if ("object" === typeof a) return a };
    l.kh = function() { return !!this.ih() };
    l.abort = function() { throw "ABORT"; };
    l.Ma = function() { return !(!0 !== this.B(S.Kc) && "true" !== this.B(S.Kc) || !this.B(S.Ba)) };
    var xz = !1,
        yz = function(a) { var b = a.eventName === S.Ac && xg() && a.Ma(),
                c = a.metadata.is_conversion || a.xb,
                d = a.metadata.is_session_start || a.s,
                e = a.metadata.create_dc_join || a.We,
                f = a.metadata.create_google_join || a.P,
                g = (a.g || a.metadata.euid_mode_enabled) && a.kh(); return !(!Hb.sendBeacon || c || g || d || e || f || b || xz) };
    var zz = !1;
    var Az = function(a) { gg("GA4_EVENT", a) };
    var Cz = function(a) { return !a || Bz.test(a) || S.Ai.hasOwnProperty(a) },
        Dz = function(a, b, c) { for (var d = c.event_param_ops || [], e = 0; e < d.length; e++) { var f = d[e]; if (f.edit_param) { var g = f.edit_param.param_name,
                        h = Rw(a, b, f.edit_param.param_value),
                        k; if (h) { var n = Number(h);
                        k = isNaN(n) ? h : n } else k = h;
                    b[g] = k } else f.delete_param && delete b[f.delete_param.param_name] } },
        Bz = /^(_|ga_|google_|gtag\.|firebase_).*$/;
    var Ez = function(a) { var b = 0,
                c = 0; return { start: function() { b = Ua() }, stop: function() { c = this.get() }, get: function() { var d = 0;
                    a.Bf() && (d = Ua() - b); return d + c } } },
        Fz = !1;
    var Gz = function() { this.g = void 0;
        this.o = 0;
        this.isActive = this.isVisible = this.s = !1;
        this.I = this.C = void 0 };
    l = Gz.prototype;
    l.Di = function(a) {
        var b = this;
        if (!this.g) {
            this.s = H.hasFocus();
            this.isVisible = !H.hidden;
            this.isActive = !0;
            var c = function(d, e, f) {
                Sb(d,
                    e,
                    function(g) { b.g.stop();
                        f(g);
                        b.Bf() && b.g.start() })
            };
            c(m, "focus", function() { b.s = !0 });
            c(m, "blur", function() { b.s = !1 });
            c(m, "pageshow", function(d) { b.isActive = !0;
                d.persisted && jg(56);
                b.I && b.I() });
            c(m, "pagehide", function() { b.isActive = !1;
                b.C && b.C() });
            c(H, "visibilitychange", function() { b.isVisible = !H.hidden });
            a.Ma() && -1 === (Hb.userAgent || "").indexOf("Firefox") && -1 === (Hb.userAgent || "").indexOf("FxiOS") && c(m, "beforeunload", function() {
                xz = !0
            });
            this.Mf();
            this.o = 0
        }
    };
    l.Mf = function() { this.o += this.ce();
        this.g = Ez(this);
        this.Bf() && this.g.start() };
    l.rk = function(a) { var b = this.ce();
        a.ab = b };
    l.yj = function(a) { Fz ? a.F[S.Bd] = void 0 : a.ab = 0;
        this.Mf();
        this.o = 0 };
    l.Bf = function() { return this.s && this.isVisible && this.isActive };
    l.tj = function() {
        return this.o +
            this.ce()
    };
    l.ce = function() { return this.g && this.g.get() || 0 };
    l.bk = function(a) { this.C = a };
    l.yh = function(a) { this.I = a };

    function Hz() { return m.gaGlobal = m.gaGlobal || {} }
    var Iz = function() { var a = Hz();
            a.hid = a.hid || Ja(); return a.hid },
        Jz = function(a, b) { var c = Hz(); if (void 0 == c.vid || b && !c.from_cookie) c.vid = a, c.from_cookie = b };
    var Lz = function(a, b, c) { if (Kz) { var d = a.metadata.client_id_source; if (void 0 === d || c <= d) a.F[S.Ha] = b, a.M[S.Ha] = void 0, a.metadata.client_id_source = c } else c <= a.I && (a.M[S.Ha] = b, a.I = c) },
        Pz = function(a, b) { var c; var d = Mz(b),
                e = String(b.B(S.na)),
                f = String(b.B(S.nb)),
                g = Number(b.B(S.za)),
                h = b.B(S.ac),
                k = { jb: S.R, domain: e, path: f, expires: g ? new Date(Ua() + 1E3 * g) : void 0, flags: "" + b.B(S.Ra) }; if (!1 === h && Nz(b) === a) c = !0;
            else { var n = cj(a, Oz[0], e, f);
                c = 1 !== Vi(d, n, k) } return c },
        Nz = function(a) {
            var b = Mz(a),
                c = "" + a.B(S.na),
                d = "" + a.B(S.nb),
                e = bj(b, c, d, Oz, S.R);
            if (!e) { var f = String(a.B(S.$b));
                f && f != b && (e = bj(f, c, d, Oz, S.R)) }
            return e
        },
        Mz = function(a) { return String(a.B(S.Ia)) + "_ga" },
        Kz = !1;
    var Oz = ["GA1"];
    var Sz = function(a, b) { var c = Qz(b),
                d = String(b.B(S.na)),
                e = String(b.B(S.nb)),
                f = Number(b.B(S.za)),
                g = cj(a, Rz[0], d, e),
                h = { jb: S.R, domain: d, path: e, expires: f ? new Date(Ua() + 1E3 * f) : void 0, flags: String(b.B(S.Ra)) }; return 1 !== Vi(c, g, h) },
        Tz = function(a) { var b = Qz(a),
                c = String(a.B(S.na)),
                d = String(a.B(S.nb)); return bj(b, c, d, Rz, S.R) },
        Uz = function(a, b, c, d, e, f, g) { if (a && b) { var h = [a, b, Oa(c), d, e]; return h.join(".") } },
        Vz = !1;
    var Rz = ["GS1"],
        Qz = function(a) { return String(a.B(S.Ia)) + "_ga_" + a.ra.substr(2) },
        Wz = function(a) { return Vz ? Uz(a.F[S.Cb], a.F[S.Sc], a.F[S.Rc], a.o, a.metadata.join_timer_sec || 0, !!a.metadata[S.Jc], a.F[S.ob]) : Uz(a.M[S.Cb], a.M[S.Sc], a.M[S.Rc], a.o, a.Nb, !!a.M[S.Jc], a.M[S.ob]) };
    var Xz = function(a) { var b = a.B(S.oa),
                c = a.getRemoteConfig(S.oa); if (c === b) return c; var d = pc(b);
            c && c[S.T] && (d[S.T] = (d[S.T] || []).concat(c[S.T])); return d },
        Yz = function(a, b) { var c = Kj(!0); return "1" !== c._up ? {} : { clientId: c[a], Bh: c[b] } },
        Zz = function(a, b, c) { var d = Kj(!0),
                e = d[b];
            e && (Lz(a, e, 2), Pz(e, a)); var f = d[c];
            f && Sz(f, a); return !(!e || !f) },
        $z = !1,
        aA = function(a) {
            var b = Xz(a) || {},
                c = Mz(a),
                d = Qz(a);
            Tj(b[S.cc], !!b[S.T]) && Zz(a, c, d) && ($z = !0);
            b[S.T] && Qj(function() {
                var e = {},
                    f = Nz(a);
                f && (e[c] = f);
                var g = Tz(a);
                g && (e[d] = g);
                var h =
                    Mi("FPLC", void 0, void 0, S.R);
                h.length && (e._fplc = h[0]);
                return e
            }, b[S.T], b[S.Nc], !!b[S.fc])
        },
        cA = function(a) { if (!a.B(S.tb)) return {}; var b = Mz(a),
                c = Qz(a);
            Rj(function() { var d; if (Kg("analytics_storage")) d = {};
                else { var e = {};
                    d = (e._up = "1", e[b] = a.M[S.Ha], e[c] = Wz(a), e) } return d }, 1); return !Kg("analytics_storage") && bA() ? Yz(b, c) : {} },
        bA = function() { var a = hi(m.location, "host"),
                b = hi(ki(H.referrer), "host"); return a && b ? a === b || 0 <= a.indexOf("." + b) || 0 <= b.indexOf("." + a) ? !0 : !1 : !1 },
        dA = function(a) {
            if (!a) return a;
            var b = String(a);
            b = Jj(b);
            return b = Jj(b, "_ga")
        };
    var eA = !1;
    var fA = function(a) { var b = !1; return b };
    var gA = function() { var a = Ua(),
            b = a + 864E5,
            c = 20,
            d = 5E3; return function() { var e = Ua();
            e >= b && (b = e + 864E5, d = 5E3); if (1 > d) return !1;
            c = Math.min(c + (e - a) / 1E3 * 5, 20);
            a = e; if (1 > c) return !1;
            d--;
            c--; return !0 } };
    var iA = function(a) { var b;
            b = hA ? a.F[S.Sa] : a.B(S.Sa); return ii(ki(b), "host", !0) },
        jA = !1;
    var hA = !1;
    var kA = !1;
    kA = !0;
    var lA = "" + Ja(),
        mA = !1,
        nA = void 0;
    var oA = function() { if (Ba(m.__uspapi)) { var a = ""; try { m.__uspapi("getUSPData", 1, function(b, c) { if (c && b) { var d = b.uspString;
                        d && RegExp("^[\\da-zA-Z-]{1,20}$").test(d) && (a = d) } }) } catch (b) {} return a } };
    var pA = function(a, b) { if (b.Ma()) { var c = oA();
                c && (a.us_privacy = c); var d = Xn();
                d && (a.gdpr = d); var e = Wn();
                e && (a.gdpr_consent = e) } },
        qA = function(a, b) { if (xg() && (a.gcs = Lg(), b.$e && (a.gcu = "1"), b.Ma())) { yg() && (a.gcd = "G1" + Fg(vg)); var c = b.B(S.fa);
                a.adr = void 0 !== c && !1 !== c ? "1" : "0";
                b.$e && (a.gcut = S.ue[b.Cg || ""]) } },
        rA = function(a, b, c) { void 0 === c && (c = {}); if ("object" === typeof b)
                for (var d in b) rA(a + "." + d, b[d], c);
            else c[a] = b; return c },
        uA = function(a) {
            var b = Pm(a.B(S.Ba), "/g/collect");
            if (b) return b;
            var c = wp(a, S.qb, a.B(S.qb));
            return c && !wp(a, S.Lc, !1) && !1 !== a.B(S.Bc) && Wp() && Kg(S.D) && Kg(S.R) ? sA() : tA()
        },
        vA = !1;
    var wA = !1;
    var xA = !1;
    var yA = !1;
    var zA = "",
        AA = !1;
    var BA = {};
    BA[S.yi] = "tid";
    BA[S.Ha] = "cid";
    BA[S.rb] = "ul";
    BA[S.Me] = "_fid";
    BA[S.Jd] = "tt";
    BA[S.Mc] = "ir";
    BA[S.Se] = "sr";
    var CA = {};
    CA[S.Cb] = "sid";
    CA[S.Sc] = "sct";
    CA[S.Rc] = "seg";
    CA[S.fb] = "dl";
    CA[S.Sa] = "dr";
    CA[S.Oc] = "dt";
    CA[S.ja] = "cu";
    CA[S.Ja] = "uid";
    var DA = {};
    var EA = {};
    EA[S.Cc] = "cc";
    EA[S.Dc] = "ci";
    EA[S.Ec] = "cm";
    EA[S.Fc] = "cn";
    EA[S.Gc] = "cs";
    EA[S.Hc] = "ck";
    var tA = function() { var a = "www";
            AA && zA && (a = zA); return "https://" + a + ".google-analytics.com/g/collect" },
        sA = function() { var a;
            AA && zA && (a = zA); return "https://" + (a ? a + "." : "") + "analytics.google.com/g/collect" },
        FA = function(a, b, c) {
            function d(D, E) { if (void 0 !== E && !S.xg.hasOwnProperty(D)) { null === E && (E = ""); var P = E;!0 === E && (P = "1");!1 === E && (P = "0"); var K; if ("_" === D.charAt(0)) e[D] = Se(P, 300);
                    else if (BA[D]) K = BA[D], e[K] = Se(P, 300);
                    else if (CA[D]) K = CA[D], g[K] = Se(P, 300);
                    else if (DA[D]) K = DA[D], f[K] = Se(P, 300);
                    else { var Q;
                        (Q = v(D, E)) || (EA[D] ? Q = !0 : D !== S.cb ? Q = !1 : ("object" !== typeof E && t(D, E), Q = !0));
                        Q || t(D, E) } } }
            var e = {},
                f = {},
                g = {};
            e.v = "2";
            e.tid = a.ra;
            e.gtm = yn();
            e._p = Iz();
            a.bf && (e.sr = a.bf);
            a.Ze && (e._z = a.Ze);
            c && (e.em = c);
            if (vA ? a.metadata.create_google_join : a.P) e._gaz = 1;
            qA(e, a);
            pA(e, a);
            a.Og && (e.gtm_up = "1");
            var h = void 0;
            wA ? h = a.F[S.zk] : h = cb(a.H.getMergedValues(S.X, 1), ".");
            h && (e.gdid = h);
            f.en = Se(a.eventName, 40);
            var k = vA ? a.metadata.is_first_visit_conversion : a.Lg;
            vA ?
                a.metadata.is_first_visit && (f._fv = k ? 2 : 1) : a.jc && (f._fv = k ? 2 : 1);
            vA ? a.metadata.is_new_to_site && (f._nsi = 1) : a.Ig && (f._nsi = 1);
            var n = vA ? a.metadata.is_session_start_conversion : a.Pg;
            vA ? a.metadata.is_session_start && (f._ss = n ? 2 : 1) : a.s && (f._ss = n ? 2 : 1);
            vA ? a.metadata.is_conversion && (f._c = 1) : a.xb && (f._c = 1);
            0 < a.ab && !vA && (f._et = a.ab);
            if (a.metadata.is_ecommerce || a.af) { var p = a.F[S.ba] || a.B(S.ba); if (Ha(p))
                    for (var q = 0; q < p.length && 200 > q; q++) f["pr" + (q + 1)] = Xe(p[q]) }
            var r = xA ? a.metadata.event_usage : a.Eg;
            r && (f._eu = r);
            var u = void 0;
            wA ? u = a.F[S.yk] : u = cb(a.H.getMergedValues(S.X, 2), ".");
            u && (f.edid = u);
            for (var t = function(D, E) { D = Se(D, 40); var F = "ep." + D,
                        P = "epn." + D;
                    D = Fa(E) ? P : F; var K = Fa(E) ? F : P;
                    f.hasOwnProperty(K) && delete f[K];
                    f[D] = Se(E, 100) }, v = function(D, E) { var F = D.split("."); if (D === S.Ca && "object" !== typeof E) return t(D, E), !0; if (F[0] === S.Ca) { if ((1 < F.length || "object" === typeof E) && a.Ma()) { var P = rA(D, E);
                            Ma(P, function(K, Q) { return void t(K, Q) }) } return !0 } return !1 }, x = 0; x < S.wg.length; ++x) { var y = S.wg[x];
                d(y, a.B(y)) }
            a.af && d(S.ja, a.B(S.ja));
            Ma(a.Ag,
                d);
            Ma(a.M, d);
            Ma(a.F, d);
            if (!yA) { var w = a.H.getKeysFromFirstOfAnyScope([S.cb, S.Cc, S.Dc, S.Ec, S.Fc, S.Gc, S.Hc]),
                    A = function(D, E) { if (void 0 !== E) { null === E && (E = ""); var F = EA[D];
                            F && (g[F] = Se(E, 300)) } };
                oc(w[S.cb]) ? Ma(w[S.cb], function(D, E) { A((S.cb + "_" + D).toLowerCase(), E) }) : Ma(w, A) }
            var B;
            B = a.F[S.Ka];
            B = B || a.B(S.Ka) || {};
            !1 !==
                a.B(S.aa) && Xp() || (B._npa = "1");
            Ma(B, function(D, E) { if (void 0 !== E && (null === E && (E = ""), b[D] !== E)) { if (D !== S.Ja || g.uid) { var F = (Fa(E) ? "upn." : "up.") + Se(D, 24);
                        f[F] = Se(E, 36) } else g.uid = Se(E, 36);
                    b[D] = E } });
            var C = !1;
            return Ze.call(this, { Ea: e, Rb: g, M: f }, uA(a), a.Ma(), C) || this
        };
    la(FA, Ze);
    var GA = function(a, b) { return a.replace(/\$\{([^\}]+)\}/g, function(c, d) { return b[d] || c }) },
        HA = function(a) { var b = a.search; return a.protocol + "//" + a.hostname + a.pathname + (b ? b + "&richsstsse" : "?richsstsse") },
        IA = function(a) { var b = {},
                c = "",
                d = a.pathname.indexOf("/g/collect");
            0 <= d && (c = a.pathname.substring(0, d));
            b.transport_url = a.protocol + "//" + a.hostname + c; return b },
        JA = function(a, b) {
            var c = new m.XMLHttpRequest;
            c.withCredentials = !0;
            var d = b ? "POST" : "GET",
                e = "",
                f = 0,
                g = ki(a),
                h = IA(g),
                k = HA(g);
            c.onprogress = function(n) {
                if (200 ===
                    c.status) {
                    e += c.responseText.substring(f);
                    f = n.loaded;
                    for (var p = GA(e, h), q = p.indexOf("\n\n"); - 1 !== q;) {
                        var r;
                        a: { var u = da(p.substring(0, q).split("\n")),
                                t = u.next().value,
                                v = u.next().value; if (t.startsWith("event: message") && v.startsWith("data: ")) try { r = JSON.parse(v.substring(v.indexOf(":") + 1)); break a } catch (P) {}
                            r = void 0 }
                        if (r) {
                            var x = r.send_pixel || [];
                            if (Array.isArray(x))
                                for (var y = 0; y < x.length; y++) {
                                    var w = x[y],
                                        A = zn('', 0);
                                    if (1 === A) {
                                        var B = Math.floor(2 * Math.random()) +
                                            1,
                                            C, D = w,
                                            E = B,
                                            F = 0 < D.indexOf("?") ? "&" : "?";
                                        C = "" + D + F + "gtmexpsb=" + E;
                                        2 === B ? Yb(C) : Rb(C)
                                    } else 2 === A ? Yb(w) : Rb(w)
                                }
                        }
                        p = p.substring(q + 2);
                        q = p.indexOf("\n\n")
                    }
                    e = p
                }
            };
            c.open(d, k);
            c.send(b)
        };
    var MA = function(a, b, c, d) { var e = a + "?" + b;
            KA && (d = !(0 === e.indexOf(tA()) || 0 === e.indexOf(sA())));
            d && !xz ? JA(e, c) : LA(a, b, c) },
        KA = !1;
    var NA = !1;
    var OA = function() { this.C = 1;
        this.I = {};
        this.g = new $e;
        this.o = -1 };
    OA.prototype.s = function(a,
        b) { var c = this,
            d; try { d = new FA(a, this.I, b) } catch (u) { a.abort() } var e = yz(a);
        e && this.g.C(d) || this.flush(); if (e && this.g.add(d)) { if (0 > this.o) { var f = m,
                    g = f.setTimeout,
                    h;
                a.Ma() ? PA ? (PA = !1, h = QA) : h = RA : h = 5E3;
                this.o = g.call(f, function() { return c.flush() }, h) } } else { var k = bf(d, this.C++);
            MA(d.o, k.If, k.body, d.C); var n = a.metadata.create_dc_join || a.We,
                p = a.metadata.create_google_join || a.P,
                q = !1 !== a.B(S.xa),
                r = !1 !== a.B(S.aa);
            SA(d, { ik: n, jk: p, wj: String(wp(a, S.Fd, a.B(S.Fd))), ef: q, Ti: r, Bj: wp(a, S.Lc, !1) }) } };
    OA.prototype.add = function(a) {
        (NA ?
            a.metadata.euid_mode_enabled : a.g) && !xz ? this.P(a) : this.s(a)
    };
    OA.prototype.flush = function() { if (this.g.events.length) { var a = cf(this.g, this.C++);
            MA(this.g.g, a.If, a.body, this.g.o);
            this.g = new $e;
            0 <= this.o && (m.clearTimeout(this.o), this.o = -1) } };
    OA.prototype.P = function(a) { var b = this,
            c = a.ih();
        c ? sh(c, function(d) { b.s(a, 1 === d.split("~").length ? void 0 : d) }) : this.s(a) };
    var SA = function(a, b) {
            function c(q) { n.push(q + "=" + encodeURIComponent("" + a.Ea[q])) } var d = b.ik,
                e = b.jk,
                f = b.wj,
                g = b.ef,
                h = b.Ti,
                k = b.Bj; if (d || e) { var n = [];
                c("tid");
                c("cid");
                c("gtm");
                n.push("aip=1");
                a.Rb.uid && n.push("uid=" + encodeURIComponent("" + a.Rb.uid));
                d && (LA("https://stats.g.doubleclick.net/g/collect", "v=2&" + n.join("&")), Eg("https://stats.g.doubleclick.net/g/collect?v=2&" + n.join("&"))); if (e) { n.push("z=" + Ja()); if (!k) { var p = f && 0 === f.indexOf("google.") && "google.com" != f ? "https://www.%/ads/ga-audiences?v=1&t=sr&slf_rd=1&_r=4&".replace("%", f) : void 0;
                        p && Rb(p + n.join("&")) } } } },
        LA = function(a, b, c) { var d = a + "?" + b;
            c ? Hb.sendBeacon && Hb.sendBeacon(d, c) : Yb(d) },
        QA = zn('', 500),
        RA = zn('', 5E3),
        PA = !0;
    var TA = window,
        UA = document,
        VA = function(a) { var b = TA._gaUserPrefs; if (b && b.ioo && b.ioo() || a && !0 === TA["ga-disable-" + a]) return !0; try { var c = TA.external; if (c && c._gaUserPrefs && "oo" == c._gaUserPrefs) return !0 } catch (f) {} for (var d = Gi("AMP_TOKEN", String(UA.cookie), !0), e = 0; e < d.length; e++)
                if ("$OPT_OUT" == d[e]) return !0;
            return UA.getElementById("__gaOptOutExtension") ? !0 : !1 };
    var WA = {};
    var XA = function(a, b, c) { c || (c = function() {}); var d = function(e) { void 0 !== e[b] && (e[b] = c(e[b])) };
            d(a.F);
            d(a.M);
            d(a.Ag);
            d(a.H.globalConfig) },
        YA = function() { var a = m.screen; return (a ? a.width : 0) + "x" + (a ? a.height : 0) },
        ZA = function(a) { var b = Mi("_opt_expid", void 0, void 0, S.R)[0]; if (b) { var c = decodeURIComponent(b).split("$"); if (3 === c.length) return c[2] } if (void 0 !== T.ga4_referrer_override) return T.ga4_referrer_override; var d = Mh("gtm.gtagReferrer." + a.ra); return d ? "" + d : H.referrer },
        $A = function() {
            var a = "",
                b = H.location;
            if (b) {
                var c =
                    b.pathname || "";
                "/" != c.charAt(0) && (c = "/" + c);
                a = b.protocol + "//" + b.hostname + c + b.search
            }
            return a
        },
        aB = function(a, b) { var c = S.D;
            Kg(c) || Ng(function() { b.$e = !0;
                b.Cg = c;
                a.Qg(b) }, c) },
        bB = {};
    WA.zi = "";
    var cB = {},
        dB = (cB[S.Cc] = !0, cB[S.Dc] = !0, cB[S.Ec] = !0, cB[S.Fc] = !0, cB[S.Gc] = !0, cB[S.Hc] = !0, cB),
        eB = function(a, b) {
            this.ab = a;
            this.jc = b;
            this.P = new OA;
            this.g = void 0;
            this.C = new Gz;
            this.o = this.s = void 0;
            this.I = !1;
            this.xb =
                void 0
        };
    l = eB.prototype;
    l.Vj = function(a, b, c) { var d = this,
            e = !1; if (c.eventModel[S.Db]) { e = !0; if ("_" === a.charAt(0)) return;
            a !== S.Ga && a !== S.Pa && Cz(a) && jg(58);
            fB(c) } var f = new wz(this.ab, a, b, c, this.jc, e),
            g = [S.R],
            h = !1;
        h = f.Ma();
        (wp(f, S.qb, f.B(S.qb)) || h) && g.push(S.D);
        Og(function() { d.Wj(f) }, g) };
    l.Wj = function(a) {
        this.o = a;
        try {
            VA(a.ra) && (jg(28), a.abort());
            0 <= WA.zi.replace(/\s+/g, "").split(",").indexOf(a.eventName) && (jg(33),
                a.abort());
            var b = a.Lb();
            b && b.blacklisted && (jg(34), a.abort());
            var c = H.location.protocol;
            "http:" != c && "https:" != c && (jg(29), a.abort());
            Hb && "preview" == Hb.loadPurpose && (jg(30), a.abort());
            var d = T.grl;
            d || (d = gA(), T.grl = d);
            d() || (jg(35), a.abort());
            gB(a);
            this.Ei(a);
            this.C.rk(a);
            var g = this.wh,
                h;
            a.B(S.tb) && !Kg(S.R) && 1 !== a.I && (a.Og = !0);
            a.eventName === S.Ga ? (a.B(S.tb) && rk(["aw", "dc"]), aA(a), h = cA(a)) : h = {};
            g.call(this, h);
            a.eventName == S.Ga && (a.Pi && 0 < a.H.getEventModelKeys().length && (jg(68), 1 < T.configCount && jg(69)), a.B(S.Aa) ||
                a.abort(), a.H.eventModel[S.X] && (a.H.containerConfig[S.X] = a.H.eventModel[S.X], a.H.eventModel[S.X] = void 0, bB[8] && (a.F[S.X] = void 0)), a.eventName = S.Ac);
            var v = this.g,
                x = this.s,
                y = this.C,
                w = Tz(a);
            w || (w = x);
            var A = Oa(a.B(S.Qc)),
                B;
            B = Oa(a.B(S.Hd));
            var C;
            a: {
                if (w) {
                    var D = w.split(".");
                    if (!(5 > D.length ||
                            7 < D.length)) { C = { sessionId: D[0], ne: Number(D[1]), od: !!Number(D[2]), Cf: Number(D[3]), Nb: Number(D[4] || 0), Ob: "1" === D[5], $d: "0" !== D[6] ? D[6] : void 0 }; break a }
                }
                C = void 0
            }
            Vz && (a.metadata.is_first_visit = !1, a.metadata.is_session_start = !1, a.metadata.join_timer_sec = 0);
            if (C && C.Nb) { var E = Math.max(0, C.Nb - Math.max(0, a.o - C.Cf));
                Vz ? a.metadata.join_timer_sec = E : a.Nb = E }
            var F = !1;
            C || (Vz ? a.metadata.is_first_visit = !0 : a.jc = !0, F = !0, C = { sessionId: String(a.o), ne: 1, od: !1, Cf: a.o, Ob: !1, $d: void 0 });
            a.o > C.Cf + 60 * A && (F = !0, C.sessionId = String(a.o), C.ne++, C.od = !1, C.$d = void 0);
            if (F) Vz ? a.metadata.is_session_start = !0 : a.s = !0, y.yj(a);
            else if (y.tj() > B || a.eventName == S.Ac) C.od = !0;
            if (Vz) { vp(a, S.Cb, C.sessionId);
                vp(a, S.Sc, C.ne);
                vp(a, S.Rc, C.od ? 1 : 0); } else {
                a.Ta(S.Cb,
                    C.sessionId);
                a.Ta(S.Sc, C.ne);
                a.Ta(S.Rc, C.od ? 1 : 0);
            }
            var X = a.B(S.Ha),
                W;
            Kz ? (a.metadata.is_new_to_site = !1, W = X ? 1 : 8) : W = a.I;
            X || (X = Nz(a), W = 3);
            X || (X = v, W = 5);
            if (!X) { var va = Kg(S.R),
                    Ea = Hz();
                X = !Ea.from_cookie || va ? Ea.vid : void 0;
                W = 6 }
            X ? X = "" + X : (X = Zi(), W = 7, Kz ? a.metadata.is_first_visit = a.metadata.is_new_to_site = !0 : a.jc = a.Ig = !0);
            Lz(a, X, W);
            iB(a);
            a: { a.Ta(S.fb, $A()); var Ca = ZA(a);Ca && a.Ta(S.Sa, Ca);a.Ta(S.Oc, H.title);a.Ta(S.rb, (Hb.language || "").toLowerCase());a.bf = YA() }
            bB[1] && (a.metadata.create_dc_join = !1, a.metadata.create_google_join = !1);
            if (!1 !== a.B(S.Bc) && Wp() && Kg(S.D)) {
                var Ga = wp(a, S.qb, a.B(S.qb));
                if ((bB[1] ? a.metadata.is_session_start : a.s) || a.B(S.gg)) bB[1] ? a.metadata.create_dc_join = !!Ga : a.We = !!Ga;
                var Ta;
                bB[1] ? Ta = a.metadata.join_timer_sec : Ta = a.Nb;
                Ga && 0 === (Ta || 0) && (bB[1] ? (a.metadata.join_timer_sec = 60, a.metadata.create_google_join = !0) : (a.Nb = 60, a.P = !0))
            }
            jB(a);
            var Tc = S.yg.hasOwnProperty(a.eventName);
            bB[2] ? Tc && (a.metadata.is_ecommerce = !0, vp(a, S.ba), vp(a, S.ja)) : a.af = Tc;
            bB[6] && vp(a, S.Jd);
            for (var Uc = a.B(S.Oe) || [], qd = 0; qd < Uc.length; qd++) {
                var Pk = Uc[qd];
                if (Pk.rule_result) {
                    bB[6] ? vp(a, S.Jd, Pk.traffic_type) :
                        a.Ta(S.Jd, Pk.traffic_type);
                    Az(3);
                    break
                }
            }
            if (a.B(S.Ba)) { var sr = Xz(a) || {},
                    tr = (Tj(sr[S.cc], !!sr[S.T]) ? Kj(!0)._fplc : void 0) || (0 < Mi("FPLC", void 0, void 0, S.R).length ? void 0 : "0");
                eA ? a.F._fplc = tr : a.M._fplc = tr }
            if (void 0 !== a.B(S.Mc)) jA && vp(a, S.Mc);
            else {
                var ur = a.B(S.Gd),
                    Qk, hh;
                a: { if ($z) { var Rk = Xz(a) || {}; if (Rk && Rk[S.T])
                            for (var vr = iA(a), ih = Rk[S.T], jf = 0; jf < ih.length; jf++)
                                if (ih[jf] instanceof RegExp) { if (ih[jf].test(vr)) { hh = !0; break a } } else if (0 <= vr.indexOf(ih[jf])) { hh = !0; break a } }
                    hh = !1 }
                var jh;
                if (!(jh = hh))
                    if (kA) jh = !1;
                    else { var oB = iA(a),
                            kh; var Sk = String(a.B(S.na)); if ("none" !== Sk)
                            if ("auto" !== Sk) kh = Sk;
                            else { if (!mA) { for (var wr = String(a.B(S.nb)), kf = Si(), ce = 0; ce < kf.length; ce++)
                                        if ("none" !== kf[ce]) { var xr = String(a.B(S.Ia)) + "_ga_autodomain"; if (0 === Vi(xr, lA, { jb: S.R, domain: kf[ce], path: wr })) { Vi(xr, void 0, { jb: S.R, domain: kf[ce], path: wr });
                                                nA = kf[ce]; break } }
                                    mA = !0 }
                                kh = nA }
                        else kh = void 0; var yr = kh;
                        jh = yr ? 0 <= oB.indexOf(yr) : !1 }
                if (!(Qk = jh)) {
                    var lh;
                    if (lh = ur) a: {
                        for (var zr = ur.include_conditions || [], pB = iA(a), Tk = 0; Tk < zr.length; Tk++)
                            if (zr[Tk].test(pB)) {
                                lh = !0;
                                break a
                            }
                        lh = !1
                    }
                    Qk = lh
                }
                Qk && (jA ? a.F[S.Mc] = "1" : a.Ta(S.Mc, 1), Az(4))
            }
            if (a.Ma() && xg()) {}
            if (a.eventName == S.Pa) { var Ar = a.B(S.eb),
                    qB = a.B(S.pb),
                    Br = void 0;
                Br = a.F[Ar];
                qB(Br || a.B(Ar));
                a.abort() }
            if (a.M[S.mc]) delete a.M[S.mc];
            else {
                var Uk = a.B(S.He);
                if (Uk) {
                    for (var Cr = Uk.edit_rules || [], Dr = !1, Vk = 0; Vk < Cr.length; Vk++) {
                        var mh;
                        a: {
                            var Db = a,
                                rd = Cr[Vk];
                            if (Uw(Db.eventName, Db.M, rd.event_name_predicate, rd.conditions || [])) {
                                if (rd.new_event_name) { var Er = Da(rd.new_event_name) ? String(rd.new_event_name) : Rw(Db.eventName, Db.M, rd.new_event_name); if (Cz(Er)) { mh = !1; break a }
                                    Db.eventName = String(Er) }
                                Dz(Db.eventName, Db.M, rd);
                                var Fr = Db.M[S.mc] = !0;
                                Az(2);
                                if (Fr) {
                                    var Gr = ig();
                                    Gr && (Db.M[S.Tc] = Gr);
                                    Uq(Db.eventName, Db.M, Db.ra);
                                    Db.abort()
                                }
                                mh = !0
                            } else mh = !1
                        }
                        mh && (Dr = !0)
                    }
                    for (var Hr = Uk.synthesis_rules || [], Wk = 0; Wk < Hr.length; Wk++) { var nh = a,
                            lf = Hr[Wk]; if (Uw(nh.eventName, nh.M, lf.event_name_predicate, lf.conditions || [])) { var Xk = lf.new_event_name; if (!Cz(Xk)) { var Yk = lf.merge_source_event_params ? pc(nh.M) : {};
                                Yk[S.mc] = !0;
                                Dz(Xk, Yk, lf);
                                Uq(Xk, Yk, nh.ra);
                                Az(1) } } }
                }
            }
            zp(a);
            var Zk = a.M[S.Tc];
            if (Ha(Zk))
                for (var $k = 0; $k < Zk.length; $k++) Az(Zk[$k]);
            var al = hg("GA4_EVENT");
            al && (zz ? a.metadata.event_usage = al : a.Eg = al);
            kB(a);
            var sB = this.wh,
                Jr = this.g,
                bl;
            var cl = Wz(a);
            cl ? (Sz(cl, a) || (jg(25), a.abort()), bl = cl) : bl = void 0;
            var tB = bl,
                oh;
            var zc;
            zc = Kz ? a.F[S.Ha] : a.M[S.Ha];
            Jr && zc === Jr ? oh = zc : zc ? (zc = "" + zc, Pz(zc, a) || (jg(31), a.abort()), Jz(zc, Kg(S.R)), oh = zc) : (jg(32), a.abort(), oh = "");
            sB.call(this, { clientId: oh, Bh: tB });
            this.pk();
            if (a.Ma()) { var uB = bB[1] ? a.metadata.is_conversion : a.xb;
                ("page_view" === a.eventName || uB) && aB(this, a) }
            this.C.Mf();
            this.xb = lB(a, this.xb);
            this.Qg(a);
            a.H.onSuccess()
        } catch (AB) { a.H.onFailure() }
        delete fg.GA4_EVENT
    };
    l.Qg = function(a) { this.P.add(a) };
    l.wh = function(a) {
        var b = a.clientId,
            c = a.Bh;
        b && c && (this.g = b, this.s = c)
    };
    l.flush = function() { this.P.flush() };
    l.pk = function() {
        var a = this;
        if (!this.I) {
            var b = Kg(S.R);
            Mg([S.R], function() {
                var c = Kg(S.R);
                if (b ^ c && a.o && a.s && a.g) {
                    var d = a.g;
                    if (c) {
                        var e = !0;
                        e = !1;
                        var f = Nz(a.o);
                        if (f) { a.g = f; var g = Tz(a.o);
                            e || g ? a.s = g : Sz(a.s, a.o) } else Pz(a.g, a.o), Jz(a.g, !0), Sz(a.s, a.o);
                        var h = {};
                        h[S.gg] = d;
                        Tq(S.ye, h, a.ab);
                    } else { a.s = void 0;
                        a.g = void 0;
                        m.gaGlobal = {}; }
                    b = c
                }
            });
            this.I = !0
        }
    };
    l.Ei = function(a) { a.eventName !== S.Pa && this.C.Di(a) };
    var hB = function(a) {
            if (a.B(S.rg)) a.g = !1, a.metadata.euid_mode_enabled = !1;
            else {
                var b = function(c) { return !!c && ("auto_detect" === c.mode || "selectors" === c.mode || "code" === c.mode) };
                bB[1] ? a.metadata.euid_mode_enabled = b(a.getRemoteConfig(S.ic)) :
                    a.g = b(a.getRemoteConfig(S.ic))
            }
        },
        gB = function(a) {},
        iB = function(a) {},
        jB = function(a) { var b = function(c) { return !!c && c.conversion };
            bB[1] ? a.metadata.is_conversion = b(a.Lb()) : a.xb = b(a.Lb()); if (bB[1] ? a.metadata.is_first_visit : a.jc) bB[1] ? a.metadata.is_first_visit_conversion = b(a.Lb("first_visit")) : a.Lg = b(a.Lb("first_visit")); if (bB[1] ? a.metadata.is_session_start : a.s) bB[1] ? a.metadata.is_session_start_conversion = b(a.Lb("session_start")) : a.Pg = b(a.Lb("session_start")) },
        kB = function(a) {
            var b = Object.keys(bB);
            if (b.length) {
                b.forEach(function(d) {
                    bB[d] &&
                        gg("CCD", Number(d))
                });
                var c = hg("CCD");
                c && (a.Ze = "ccd." + c)
            }
        },
        lB = function(a, b) { var c = void 0; return c };

    function fB(a) { delete a.eventModel[S.Db];
        mB(a.eventModel) }
    var mB = function(a) { Ma(a, function(c) { "_" === c.charAt(0) && delete a[c] }); var b = a[S.Ka] || {};
        Ma(b, function(c) { "_" === c.charAt(0) && delete b[c] }) };
    var nB = function(a) { if ("prerender" == H.visibilityState) return !1;
            a(); return !0 },
        rB = function(a) { if (!nB(a)) { var b = !1,
                    c = function() {!b && nB(a) && (b = !0, Tb(H, "visibilitychange", c), jg(55)) };
                Sb(H, "visibilitychange", c);
                jg(54) } };
    var vB = function(a, b, c) { Tq(b, c, a) },
        wB = function(a, b, c) { Tq(b, c, a, !0) },
        yB = function(a, b) { var c = new eB(a, b);
            rB(function() { xB(a, c) }); };

    function xB(a, b) { var c = b.C;
        Sq(a, function(d, e, f, g) { b.Vj(e, f, g) });
        c.bk(function() { Qq().flush();
            1E3 <= c.ce() && Hb.sendBeacon && vB(a, S.ye, {});
            xz = !0;
            b.flush();
            c.yh(function() { xz = !1;
                c.yh(void 0) }) }); }
    var Z = { h: {} };


    Z.h.ehl = ["google"],
        function() {
            function a(k) { return k.target && k.target.location && k.target.location.href ? k.target.location.href : Jt() }

            function b(k, n) { Sb(k, "hashchange", function(p) { var q = a(p);
                    n({ source: "hashchange", state: null, url: Lt(q), N: Kt(q) }) }) }

            function c(k, n) { Sb(k, "popstate", function(p) { var q = a(p);
                    n({ source: "popstate", state: p.state, url: Lt(q), N: Kt(q) }) }) }

            function d(k, n, p) {
                var q = n.history,
                    r = q[k];
                if (Ba(r)) try {
                    q[k] = function(u, t, v) {
                        r.apply(q, [].slice.call(arguments, 0));
                        p({
                            source: k,
                            state: u,
                            url: Lt(Jt()),
                            N: Kt(Jt())
                        })
                    }
                } catch (u) {}
            }

            function e() { var k = { source: null, state: V("history").state || null, url: Lt(Jt()), N: Kt(Jt()) }; return function(n, p) { var q = k,
                        r = {};
                    r[q.source] = !0;
                    r[n.source] = !0; if (!r.popstate || !r.hashchange || q.N != n.N) { var u = { event: "gtm.historyChange-v2", "gtm.historyChangeSource": n.source, "gtm.oldUrlFragment": k.N, "gtm.newUrlFragment": n.N, "gtm.oldHistoryState": k.state, "gtm.newHistoryState": n.state, "gtm.oldUrl": k.url, "gtm.newUrl": n.url, "gtm.triggers": p.join(",") };
                        k = n;
                        Nt(u) } } }

            function f(k, n) {
                var p = "" +
                    n;
                if (g[p]) g[p].push(k);
                else { var q = [k];
                    g[p] = q; var r = e(),
                        u = -1;
                    h.push(function(t) { 0 <= u && Gt(u);
                        n ? u = Ht(function() { r(t, q);
                            u = -1 }, n) : r(t, q) }) }
            }
            var g = {},
                h = [];
            (function(k) { Z.__ehl = k;
                Z.__ehl.m = "ehl";
                Z.__ehl.isVendorTemplate = !0;
                Z.__ehl.priorityOverride = 0 })(function(k) {
                var n = V("self"),
                    p = k.vtp_uniqueTriggerId || "0",
                    q = k.vtp_groupEvents ? Number(k.vtp_groupEventsInterval) : 0;
                0 > q ? q = 0 : isNaN(q) && (q = 1E3);
                if (St("ehl")) {
                    var r = lt("ehl", "reg");
                    r ? (r(p, q),
                        I(k.vtp_gtmOnSuccess)) : I(k.vtp_gtmOnFailure)
                } else { var u = function(t) { for (var v = 0; v < h.length; v++) h[v](t) };
                    b(n, u);
                    c(n, u);
                    d("pushState", n, u);
                    d("replaceState", n, u);
                    f(p, q);
                    jt("ehl", "reg", f);
                    Tt("ehl");
                    I(k.vtp_gtmOnSuccess) }
            })
        }();
    Z.h.sdl = ["google"],
        function() {
            function a() { return !!(Object.keys(k("horiz.pix")).length || Object.keys(k("horiz.pct")).length || Object.keys(k("vert.pix")).length || Object.keys(k("vert.pct")).length) }

            function b(w) { for (var A = [], B = w.split(","), C = 0; C < B.length; C++) { var D = Number(B[C]); if (isNaN(D)) return [];
                    p.test(B[C]) || A.push(D) } return A }

            function c() { var w = 0,
                    A = 0; return function() { var B = Xh(),
                        C = B.height;
                    w = Math.max(v.scrollLeft + B.width, w);
                    A = Math.max(v.scrollTop + C, A); return { qf: w, rf: A } } }

            function d() {
                u = V("self");
                t = u.document;
                v = t.scrollingElement || t.body && t.body.parentNode;
                y = c()
            }

            function e(w, A, B, C) { var D = k(A),
                    E = {},
                    F; for (F in D) { E.Vb = F; if (D.hasOwnProperty(E.Vb)) { var P = Number(E.Vb);
                        w < P || (Nt({ event: "gtm.scrollDepth", "gtm.scrollThreshold": P, "gtm.scrollUnits": B.toLowerCase(), "gtm.scrollDirection": C, "gtm.triggers": D[E.Vb].join(",") }), kt("sdl", A, function(K) { return function(Q) { delete Q[K.Vb]; return Q } }(E), {})) }
                    E = { Vb: E.Vb } } }

            function f() {
                var w = y(),
                    A = w.qf,
                    B = w.rf,
                    C = A / v.scrollWidth * 100,
                    D = B / v.scrollHeight * 100;
                e(A, "horiz.pix",
                    q.Nd, r.zg);
                e(C, "horiz.pct", q.Md, r.zg);
                e(B, "vert.pix", q.Nd, r.Ng);
                e(D, "vert.pct", q.Md, r.Ng);
                jt("sdl", "pending", !1)
            }

            function g() { var w = 250,
                    A = !1;
                t.scrollingElement && t.documentElement && u.addEventListener && (w = 50, A = !0); var B = 0,
                    C = !1,
                    D = function() { C ? B = Ht(D, w) : (B = 0, f(), St("sdl") && !a() && (Tb(u, "scroll", E), Tb(u, "resize", E), jt("sdl", "init", !1)));
                        C = !1 },
                    E = function() { A && y();
                        B ? C = !0 : (B = Ht(D, w), jt("sdl", "pending", !0)) }; return E }

            function h(w, A, B) {
                if (A) {
                    var C = b(String(w));
                    kt("sdl", B, function(D) {
                        for (var E = 0; E < C.length; E++) {
                            var F =
                                String(C[E]);
                            D.hasOwnProperty(F) || (D[F] = []);
                            D[F].push(A)
                        }
                        return D
                    }, {})
                }
            }

            function k(w) { return lt("sdl", w, {}) }

            function n(w) {
                I(w.vtp_gtmOnSuccess);
                var A = w.vtp_uniqueTriggerId,
                    B = w.vtp_horizontalThresholdsPixels,
                    C = w.vtp_horizontalThresholdsPercent,
                    D = w.vtp_verticalThresholdUnits,
                    E = w.vtp_verticalThresholdsPixels,
                    F = w.vtp_verticalThresholdsPercent;
                switch (w.vtp_horizontalThresholdUnits) {
                    case q.Nd:
                        h(B, A, "horiz.pix"); break;
                    case q.Md:
                        h(C, A, "horiz.pct") }
                switch (D) {
                    case q.Nd:
                        h(E, A, "vert.pix");
                        break;
                    case q.Md:
                        h(F,
                            A, "vert.pct")
                }
                St("sdl") ? lt("sdl", "pending") || (x || (d(), x = !0), I(function() { return f() })) : (d(), x = !0, v && (Tt("sdl"), jt("sdl", "pending", !0), I(function() { f(); if (a()) { var P = g();
                        Sb(u, "scroll", P);
                        Sb(u, "resize", P) } else jt("sdl", "init", !1) })))
            }
            var p = /^\s*$/,
                q = { Md: "PERCENT", Nd: "PIXELS" },
                r = { Ng: "vertical", zg: "horizontal" },
                u, t, v, x = !1,
                y;
            (function(w) { Z.__sdl = w;
                Z.__sdl.m = "sdl";
                Z.__sdl.isVendorTemplate = !0;
                Z.__sdl.priorityOverride = 0 })(function(w) { w.vtp_triggerStartOption ? n(w) : ss(function() { n(w) }) })
        }();



    Z.h.read_container_data = ["google"],
        function() {
            (function(a) { Z.__read_container_data = a;
                Z.__read_container_data.m = "read_container_data";
                Z.__read_container_data.isVendorTemplate = !0;
                Z.__read_container_data.priorityOverride = 0 })(function() { return { assert: function() {}, V: function() { return {} } } }) }();
    Z.h.c = ["google"],
        function() {
            (function(a) { Z.__c = a;
                Z.__c.m = "c";
                Z.__c.isVendorTemplate = !0;
                Z.__c.priorityOverride = 0 })(function(a) { Ut(a.vtp_value, "c", a.vtp_gtmEventId); return a.vtp_value }) }();
    Z.h.e = ["google"],
        function() {
            (function(a) { Z.__e = a;
                Z.__e.m = "e";
                Z.__e.isVendorTemplate = !0;
                Z.__e.priorityOverride = 0 })(function(a) { return String(a.vtp_gtmCachedValues.event) }) }();

    Z.h.u = ["google"],
        function() {
            var a = function(b) { return { toString: function() { return b } } };
            (function(b) { Z.__u = b;
                Z.__u.m = "u";
                Z.__u.isVendorTemplate = !0;
                Z.__u.priorityOverride = 0 })(function(b) {
                var c;
                c = (c = b.vtp_customUrlSource ? b.vtp_customUrlSource : Mt("gtm.url", 1)) || Jt();
                var d = b[a("vtp_component")];
                if (!d || "URL" == d) return Lt(String(c));
                var e = ki(String(c)),
                    f;
                if ("QUERY" === d) a: {
                    var g = b[a("vtp_multiQueryKeys").toString()],
                        h = b[a("vtp_queryKey").toString()] || "",
                        k = b[a("vtp_ignoreEmptyQueryParam").toString()],
                        n;g ? Ha(h) ?
                    n = h : n = String(h).replace(/\s+/g, "").split(",") : n = [String(h)];
                    for (var p = 0; p < n.length; p++) { var q = ii(e, "QUERY", void 0, void 0, n[p]); if (void 0 != q && (!k || "" !== q)) { f = q; break a } }
                    f = void 0
                }
                else f = ii(e, d, "HOST" == d ? b[a("vtp_stripWww")] : void 0, "PATH" == d ? b[a("vtp_defaultPages")] : void 0, void 0);
                return f
            })
        }();
    Z.h.v = ["google"],
        function() {
            (function(a) { Z.__v = a;
                Z.__v.m = "v";
                Z.__v.isVendorTemplate = !0;
                Z.__v.priorityOverride = 0 })(function(a) { var b = a.vtp_name; if (!b || !b.replace) return !1; var c = Mt(b.replace(/\\\./g, "."), a.vtp_dataLayerVersion || 1),
                    d = void 0 !== c ? c : a.vtp_defaultValue;
                Ut(d, "v", a.vtp_gtmEventId); return d }) }();

    Z.h.ytl = ["google"],
        function() {
            function a() { var t = Math.round(1E9 * Math.random()) + ""; return H.getElementById(t) ? a() : t }

            function b(t, v) { if (!t) return !1; for (var x = 0; x < p.length; x++)
                    if (0 <= t.indexOf("//" + p[x] + "/" + v)) return !0;
                return !1 }

            function c(t, v) {
                var x = t.getAttribute("src");
                if (b(x, "embed/")) {
                    if (0 < x.indexOf("enablejsapi=1")) return !0;
                    if (v) {
                        var y = t.setAttribute,
                            w;
                        var A = -1 !== x.indexOf("?") ? "&" : "?";
                        if (-1 < x.indexOf("origin=")) w = x + A + "enablejsapi=1";
                        else {
                            if (!r) {
                                var B = V("document");
                                r = B.location.protocol + "//" + B.location.hostname;
                                B.location.port && (r += ":" + B.location.port)
                            }
                            w = x + A + "enablejsapi=1&origin=" + encodeURIComponent(r)
                        }
                        y.call(t, "src", w);
                        return !0
                    }
                }
                return !1
            }

            function d(t, v) { if (!t.getAttribute("data-gtm-yt-inspected-" + v.Ya) && (t.setAttribute("data-gtm-yt-inspected-" + v.Ya, "true"), c(t, v.cd))) { t.id || (t.id = a()); var x = V("YT"),
                        y = x.get(t.id);
                    y || (y = new x.Player(t.id)); var w = f(y, v),
                        A = {},
                        B; for (B in w) A.zc = B, w.hasOwnProperty(A.zc) && y.addEventListener(A.zc, function(C) { return function(D) { return w[C.zc](D.data) } }(A)), A = { zc: A.zc } } }

            function e(t) {
                I(function() {
                    function v() {
                        for (var y =
                                x.getElementsByTagName("iframe"), w = y.length, A = 0; A < w; A++) d(y[A], t)
                    }
                    var x = V("document");
                    v();
                    ut(v)
                })
            }

            function f(t, v) {
                var x, y;

                function w() { Q = Ft(function() { return { url: L, title: ba, sf: O, $g: t.getCurrentTime(), playbackRate: X } }, v.Ya, t.getIframe());
                    O = 0;
                    ba = L = "";
                    X = 1; return A }

                function A(Ca) {
                    switch (Ca) {
                        case q.PLAYING:
                            O = Math.round(t.getDuration());
                            L = t.getVideoUrl();
                            if (t.getVideoData) { var Ga = t.getVideoData();
                                ba = Ga ? Ga.title : "" }
                            X = t.getPlaybackRate();
                            v.kf ? Nt(Q.createEvent("start")) : Q.nc();
                            R = k(v.Kf, v.Jf, t.getDuration());
                            return B(Ca);
                        default:
                            return A
                    }
                }

                function B() { W = t.getCurrentTime();
                    va = It().getTime();
                    Q.Ah();
                    K(); return C }

                function C(Ca) { var Ga; switch (Ca) {
                        case q.ENDED:
                            return E(Ca);
                        case q.PAUSED:
                            Ga = "pause";
                        case q.BUFFERING:
                            var Ta = t.getCurrentTime() - W;
                            Ga = 1 < Math.abs((It().getTime() - va) / 1E3 * X - Ta) ? "seek" : Ga || "buffering";
                            t.getCurrentTime() && (v.jf ? Nt(Q.createEvent(Ga)) : Q.nc());
                            P(); return D;
                        case q.UNSTARTED:
                            return w(Ca);
                        default:
                            return C } }

                function D(Ca) {
                    switch (Ca) {
                        case q.ENDED:
                            return E(Ca);
                        case q.PLAYING:
                            return B(Ca);
                        case q.UNSTARTED:
                            return w(Ca);
                        default:
                            return D
                    }
                }

                function E() { for (; y;) { var Ca = x;
                        Gt(y);
                        Ca() }
                    v.hf && Nt(Q.createEvent("complete", 1)); return w(q.UNSTARTED) }

                function F() {}

                function P() { y && (Gt(y), y = 0, x = F) }

                function K() { if (R.length && 0 !== X) { var Ca = -1,
                            Ga;
                        do { Ga = R[0]; if (Ga.ka > t.getDuration()) return;
                            Ca = (Ga.ka - t.getCurrentTime()) / X; if (0 > Ca && (R.shift(), 0 === R.length)) return } while (0 > Ca);
                        x = function() { y = 0;
                            x = F;
                            0 < R.length && R[0].ka === Ga.ka && (R.shift(), Nt(Q.createEvent("progress", Ga.kd, Ga.nd)));
                            K() };
                        y = Ht(x, 1E3 * Ca) } }
                var Q, R = [],
                    O, L, ba, X, W, va, Ea = w(q.UNSTARTED);
                y = 0;
                x = F;
                return { onStateChange: function(Ca) { Ea = Ea(Ca) }, onPlaybackRateChange: function(Ca) { W = t.getCurrentTime();
                        va = It().getTime();
                        Q.nc();
                        X = Ca;
                        P();
                        K() } }
            }

            function g(t) { for (var v = t.split(","), x = v.length, y = [], w = 0; w < x; w++) { var A = parseInt(v[w], 10);
                    isNaN(A) || 100 < A || 0 > A || y.push(A / 100) }
                y.sort(function(B, C) { return B - C }); return y }

            function h(t) { for (var v = t.split(","), x = v.length, y = [], w = 0; w < x; w++) { var A = parseInt(v[w], 10);
                    isNaN(A) || 0 > A || y.push(A) }
                y.sort(function(B, C) { return B - C }); return y }

            function k(t, v, x) {
                var y = t.map(function(B) {
                    return {
                        ka: B,
                        nd: B,
                        kd: void 0
                    }
                });
                if (!v.length) return y;
                var w = v.map(function(B) { return { ka: B * x, nd: void 0, kd: B } });
                if (!y.length) return w;
                var A = y.concat(w);
                A.sort(function(B, C) { return B.ka - C.ka });
                return A
            }

            function n(t) {
                var v = !!t.vtp_captureStart,
                    x = !!t.vtp_captureComplete,
                    y = !!t.vtp_capturePause,
                    w = g(t.vtp_progressThresholdsPercent + ""),
                    A = h(t.vtp_progressThresholdsTimeInSeconds + ""),
                    B = !!t.vtp_fixMissingApi;
                if (v || x || y || w.length || A.length) {
                    var C = { kf: v, hf: x, jf: y, Jf: w, Kf: A, cd: B, Ya: void 0 === t.vtp_uniqueTriggerId ? "" : t.vtp_uniqueTriggerId },
                        D = V("YT"),
                        E = function() { e(C) };
                    I(t.vtp_gtmOnSuccess);
                    if (D) D.ready && D.ready(E);
                    else { var F = V("onYouTubeIframeAPIReady");
                        Ot("onYouTubeIframeAPIReady", function() { F && F();
                            E() });
                        I(function() { for (var P = V("document"), K = P.getElementsByTagName("script"), Q = K.length, R = 0; R < Q; R++) { var O = K[R].getAttribute("src"); if (b(O, "iframe_api") || b(O, "player_api")) return } for (var L = P.getElementsByTagName("iframe"), ba = L.length, X = 0; X < ba; X++)
                                if (!u && c(L[X], C.cd)) { U("https://www.youtube.com/iframe_api");
                                    u = !0; break } }) }
                } else I(t.vtp_gtmOnSuccess)
            }
            var p = ["www.youtube.com", "www.youtube-nocookie.com"],
                q = { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
                r, u = !1;
            (function(t) { Z.__ytl = t;
                Z.__ytl.m = "ytl";
                Z.__ytl.isVendorTemplate = !0;
                Z.__ytl.priorityOverride = 0 })(function(t) { t.vtp_triggerStartOption ? n(t) : nl(function() { n(t) }) })
        }();

    Z.h.aev = ["google"],
        function() {
            function a(r, u, t, v, x) { x || (x = "element"); var y = u + "." + t,
                    w; if (n.hasOwnProperty(y)) w = n[y];
                else { var A = r[x]; if (A && (w = v(A), n[y] = w, p.push(y), 35 < p.length)) { var B = p.shift();
                        delete n[B] } } return w }

            function b(r, u, t) { var v = r[q[u]]; return void 0 !== v ? v : t }

            function c(r, u) {
                if (!r) return !1;
                var t = d(Jt());
                Ha(u) || (u = String(u || "").replace(/\s+/g, "").split(","));
                for (var v = [t], x = 0; x < u.length; x++) {
                    var y = u[x];
                    if (y.hasOwnProperty("is_regex"))
                        if (y.is_regex) try { y = new RegExp(y.domain) } catch (B) { continue } else y =
                            y.domain;
                    var w = d(r);
                    if (y instanceof RegExp) { if (y.test(w)) return !1 } else { var A = y; if (0 != A.length) { if (0 <= w.indexOf(A)) return !1;
                            v.push(d(A)) } }
                }
                return !jy(r, v)
            }

            function d(r) { k.test(r) || (r = "http://" + r); return ii(ki(r), "HOST", !0) }

            function e(r, u, t, v) {
                switch (r) {
                    case "SUBMIT_TEXT":
                        return a(u, t, "FORM." + r, f, "formSubmitElement") || v;
                    case "LENGTH":
                        var x = a(u, t, "FORM." + r, g);
                        return void 0 === x ? v : x;
                    case "INTERACTED_FIELD_ID":
                        return h(u, "id", v);
                    case "INTERACTED_FIELD_NAME":
                        return h(u, "name", v);
                    case "INTERACTED_FIELD_TYPE":
                        return h(u,
                            "type", v);
                    case "INTERACTED_FIELD_POSITION":
                        var y = u.interactedFormFieldPosition;
                        return void 0 === y ? v : y;
                    case "INTERACT_SEQUENCE_NUMBER":
                        var w = u.interactSequenceNumber;
                        return void 0 === w ? v : w;
                    default:
                        return v
                }
            }

            function f(r) { switch (r.tagName.toLowerCase()) {
                    case "input":
                        return Ub(r, "value");
                    case "button":
                        return Vb(r);
                    default:
                        return null } }

            function g(r) { if ("form" === r.tagName.toLowerCase() && r.elements) { for (var u = 0, t = 0; t < r.elements.length; t++) pt(r.elements[t]) && u++; return u } }

            function h(r, u, t) {
                var v = r.interactedFormField;
                return v && Ub(v, u) || t
            }
            var k = /^https?:\/\//i,
                n = {},
                p = [],
                q = { ATTRIBUTE: "elementAttribute", CLASSES: "elementClasses", ELEMENT: "element", ID: "elementId", HISTORY_CHANGE_SOURCE: "historyChangeSource", HISTORY_NEW_STATE: "newHistoryState", HISTORY_NEW_URL_FRAGMENT: "newUrlFragment", HISTORY_OLD_STATE: "oldHistoryState", HISTORY_OLD_URL_FRAGMENT: "oldUrlFragment", TARGET: "elementTarget" };
            (function(r) { Z.__aev = r;
                Z.__aev.m = "aev";
                Z.__aev.isVendorTemplate = !0;
                Z.__aev.priorityOverride = 0 })(function(r) {
                var u = r.vtp_gtmEventId,
                    t = r.vtp_defaultValue,
                    v = r.vtp_varType,
                    x = r.vtp_gtmCachedValues.gtm;
                switch (v) {
                    case "TAG_NAME":
                        var y = x.element;
                        return y && y.tagName || t;
                    case "TEXT":
                        return a(x, u, v, Vb) || t;
                    case "URL":
                        var w;
                        a: { var A = String(x.elementUrl || t || ""),
                                B = ki(A),
                                C = String(r.vtp_component || "URL"); switch (C) {
                                case "URL":
                                    w = A; break a;
                                case "IS_OUTBOUND":
                                    w = c(A, r.vtp_affiliatedDomains); break a;
                                default:
                                    w = ii(B, C, r.vtp_stripWww, r.vtp_defaultPages, r.vtp_queryKey) } }
                        return w;
                    case "ATTRIBUTE":
                        var D;
                        if (void 0 === r.vtp_attribute) D = b(x, v, t);
                        else {
                            var E = x.element;
                            D = E && Ub(E, r.vtp_attribute) ||
                                t || ""
                        }
                        return D;
                    case "MD":
                        var F = r.vtp_mdValue,
                            P = a(x, u, "MD", Bt);
                        return F && P ? Et(P, F) || t : P || t;
                    case "FORM":
                        return e(String(r.vtp_component || "SUBMIT_TEXT"), x, u, t);
                    default:
                        var K = b(x, v, t);
                        Ut(K, "aev", r.vtp_gtmEventId);
                        return K
                }
            })
        }();
    Z.h.dlm = ["google"],
        function() {
            (function(a) { Z.__dlm = a;
                Z.__dlm.m = "dlm";
                Z.__dlm.isVendorTemplate = !0;
                Z.__dlm.priorityOverride = 0 })(function(a) { var b = ky(a.vtp_userInput || [], "key", "value") || {};
                a.vtp_synchronousWrite ? Ma(b, function(c, d) { Ph(c, d) }) : Nt(b);
                I(a.vtp_gtmOnSuccess) }) }();

    Z.h.gct = ["google"],
        function() {
            function a(d) { for (var e = [], f = 0; f < d.length; f++) try { e.push(new RegExp(d[f])) } catch (g) {}
                return e }

            function b(d) { return d.replace(/[.*+\-?^${}()|[\]\\]/g, "\\$&") }

            function c(d) { for (var e = [], f = 0; f < d.length; f++) { var g = d[f].matchValue,
                        h; switch (d[f].matchType) {
                        case "BEGINS_WITH":
                            h = "^" + b(g); break;
                        case "ENDS_WITH":
                            h = b(g) + "$"; break;
                        case "EQUALS":
                            h = "^" + b(g) + "$"; break;
                        case "REGEX":
                            h = g; break;
                        default:
                            h = b(g) }
                    e.push(h) } return e }(function(d) {
                Z.__gct = d;
                Z.__gct.m = "gct";
                Z.__gct.isVendorTemplate = !0;
                Z.__gct.priorityOverride = 0
            })(function(d) {
                var e = {};
                0 < d.vtp_sessionDuration && (e[S.Qc] = d.vtp_sessionDuration);
                e[S.Cd] = d.vtp_eventSettings;
                e[S.He] = d.vtp_dynamicEventSettings;
                e[S.qb] = 1 === d.vtp_googleSignals;
                e[S.Fd] = d.vtp_foreignTld;
                e[S.Lc] = 1 === d.vtp_restrictDomain;
                e[S.Oe] = d.vtp_internalTrafficResults;
                var f = S.oa,
                    g = d.vtp_linker;
                g && g[S.T] && (g[S.T] = a(g[S.T]));
                e[f] = g;
                var h = S.Gd,
                    k = d.vtp_referralExclusionDefinition;
                k && k.include_conditions && (k.include_conditions = a(k.include_conditions));
                e[h] = k;
                var n = Xq(d.vtp_trackingId),
                    p = n.referral_exclusion_conditions;
                p && (p.length && "object" === typeof p[0] && (p = c(p)), e[S.Gd] = { include_conditions: a(p) });
                var q = n.cross_domain_conditions;
                if (q) { q.length && "object" === typeof q[0] && (q = c(q)); var r = {};
                    e[S.oa] = (r[S.T] = a(q), r[S.fc] = !0, r[S.cc] = !0, r[S.Nc] = "query", r) }
                yB(d.vtp_trackingId, e);
                I(d.vtp_gtmOnSuccess)
            })
        }();



    Z.h.get = ["google"],
        function() {
            (function(a) { Z.__get = a;
                Z.__get.m = "get";
                Z.__get.isVendorTemplate = !0;
                Z.__get.priorityOverride = 0 })(function(a) { var b = a.vtp_settings;
                (a.vtp_deferrable ? wB : vB)(String(b.streamId), String(a.vtp_eventName), b.eventParameters || {});
                a.vtp_gtmOnSuccess() }) }();








    Z.h.lcl = [],
        function() {
            function a() {
                var c = V("document"),
                    d = 0,
                    e = function(f) {
                        var g = f.target;
                        if (g && 3 !== f.which && !(f.Af || f.timeStamp && f.timeStamp === d)) {
                            d = f.timeStamp;
                            g = Xb(g, ["a", "area"], 100);
                            if (!g) return f.returnValue;
                            var h = f.defaultPrevented || !1 === f.returnValue,
                                k = lt("lcl", h ? "nv.mwt" : "mwt", 0),
                                n;
                            n = h ? lt("lcl", "nv.ids", []) : lt("lcl", "ids", []);
                            if (n.length) {
                                var p = ht(g, "gtm.linkClick", n);
                                if (b(f, g, c) && !h && k && g.href) {
                                    var q = !!Ia(String(Zb(g, "rel") || "").split(" "), function(v) { return "noreferrer" === v.toLowerCase() });
                                    q && jg(36);
                                    var r = V((Zb(g, "target") || "_self").substring(1)),
                                        u = !0,
                                        t = Ts(function() { var v; if (v = u && r) { var x;
                                                a: if (q) { var y; try { y = new MouseEvent(f.type, { bubbles: !0 }) } catch (w) { if (!c.createEvent) { x = !1; break a }
                                                        y = c.createEvent("MouseEvents");
                                                        y.initEvent(f.type, !0, !0) }
                                                    y.Af = !0;
                                                    f.target.dispatchEvent(y);
                                                    x = !0 } else x = !1;
                                                v = !x }
                                            v && (r.location.href = Zb(g, "href")) }, k);
                                    if (Nt(p, t, k)) u = !1;
                                    else return f.preventDefault && f.preventDefault(), f.returnValue = !1
                                } else Nt(p, function() {}, k || 2E3);
                                return !0
                            }
                        }
                    };
                Sb(c, "click", e, !1);
                Sb(c, "auxclick",
                    e, !1)
            }

            function b(c, d, e) { if (2 === c.which || c.ctrlKey || c.shiftKey || c.altKey || c.metaKey) return !1; var f = Zb(d, "href"),
                    g = f.indexOf("#"),
                    h = Zb(d, "target"); if (h && "_self" !== h && "_parent" !== h && "_top" !== h || 0 === g) return !1; if (0 < g) { var k = Lt(f),
                        n = Lt(e.location); return k !== n } return !0 }(function(c) { Z.__lcl = c;
                Z.__lcl.m = "lcl";
                Z.__lcl.isVendorTemplate = !0;
                Z.__lcl.priorityOverride = 0 })(function(c) {
                var d = void 0 === c.vtp_waitForTags ? !0 : c.vtp_waitForTags,
                    e = void 0 === c.vtp_checkValidation ? !0 : c.vtp_checkValidation,
                    f = Number(c.vtp_waitForTagsTimeout);
                if (!f || 0 >= f) f = 2E3;
                var g = c.vtp_uniqueTriggerId || "0";
                if (d) { var h = function(n) { return Math.max(f, n) };
                    kt("lcl", "mwt", h, 0);
                    e || kt("lcl", "nv.mwt", h, 0) }
                var k = function(n) { n.push(g); return n };
                kt("lcl", "ids", k, []);
                e || kt("lcl", "nv.ids", k, []);
                St("lcl") || (a(), Tt("lcl"));
                I(c.vtp_gtmOnSuccess)
            })
        }();
    var zB = {};
    zB.dataLayer = Nh;
    zB.callback = function(a) { Eh.hasOwnProperty(a) && Ba(Eh[a]) && Eh[a]();
        delete Eh[a] };
    zB.bootstrap = 0;
    zB._spx = !1;
    (function(a) {
        if (!m["__TAGGY_INSTALLED"]) { var b = !1; if (H.referrer) { var c = ki(H.referrer);
                b = "cct.google" === hi(c, "host") } if (!b) { var d = Mi("googTaggyReferrer");
                b = d.length && d[0].length }
            b && (m["__TAGGY_INSTALLED"] = !0, Ob("https://cct.google/taggy/agent.js")) }
        var f = function(q) {
                var r = "GTM",
                    u = "GTM";
                r = "OGT", u = "GTAG";
                var t = m["google.tagmanager.debugui2.queue"];
                t || (t = [], m["google.tagmanager.debugui2.queue"] = t, Ob("https://" + J.yd + "/debug/bootstrap?id=" + J.J + "&src=" + u + "&cond=" + q + "&gtm=" + yn()));
                var v = { messageType: "CONTAINER_STARTING", data: { scriptSource: Ib, containerProduct: r, debug: !1, id: J.J } };
                v.data.resume = function() { a() };
                J.Rh && (v.data.initialPublish = !0);
                t.push(v)
            },
            g = void 0,
            h = ii(m.location, "query", !1, void 0, "gtm_debug");
        bt(h) && (g = 2);
        if (!g && H.referrer) { var k = ki(H.referrer); "tagassistant.google.com" === hi(k, "host") && (g = 3) }
        if (!g) { var n = Mi("__TAG_ASSISTANT");
            n.length && n[0].length && (g = 4) }
        if (!g) { var p = H.documentElement.getAttribute("data-tag-assistant-present");
            bt(p) && (g = 5) }
        g && Ib ? f(g) : a()
    })(function() {
        var a = !1;
        a && wl("INIT");
        mg().o();
        Vn();
        Uj.enable_gbraid_cookie_write = !0;
        var b = !!T[J.J];
        if (b) { var c = T.zones;
            c && c.unregisterChild(J.J); } else {
            for (var g = data.resource || {}, h = g.macros || [], k = 0; k < h.length; k++) ie.push(h[k]);
            for (var n = g.tags || [], p = 0; p < n.length; p++) le.push(n[p]);
            for (var q = g.predicates || [], r = 0; r < q.length; r++) ke.push(q[r]);
            for (var u = g.rules || [], t = 0; t < u.length; t++) { for (var v = u[t], x = {}, y = 0; y < v.length; y++) x[v[y][0]] = Array.prototype.slice.call(v[y], 1);
                je.push(x) }
            ne = Z;
            oe = wu;
            Le = new Ke;
            var w = data.sandboxed_scripts,
                A = data.security_groups,
                B = data.runtime || [],
                C = data.runtime_lines;
            cy = new Id;
            fy();
            he = dy();
            var D = cy,
                E = $x();
            lb(D.g, "require", E);
            for (var F = 0; F < B.length; F++) { var P = B[F]; if (!Ha(P) || 3 > P.length) { if (0 === P.length) continue; break }
                C && C[F] && C[F].length && xe(P, C[F]);
                cy.execute(P) }
            if (void 0 !==
                w)
                for (var K = ["sandboxedScripts"], Q = 0; Q < w.length; Q++) { var R = w[Q].replace(/^_*/, "");
                    Fh[R] = K }
            gy(A);
            T[J.J] = zB;
            Xa(Fh, Z.h);
            qe = Ee;
            at();
            il = !1;
            jl = 0;
            if ("interactive" == H.readyState && !H.createEventObject || "complete" == H.readyState) ll();
            else {
                Sb(H, "DOMContentLoaded",
                    ll);
                Sb(H, "readystatechange", ll);
                if (H.createEventObject && H.documentElement.doScroll) { var O = !0; try { O = !m.frameElement } catch (va) {}
                    O && ml() }
                Sb(m, "load", ll)
            }
            ps = !1;
            "complete" === H.readyState ? rs() : Sb(m, "load", rs);
            um && m.setInterval(om, 864E5);
            Ch = Ua();
            zB.bootstrap = Ch;
            if (a) {
                var W = xl("INIT");
            }
        }
    });

})()