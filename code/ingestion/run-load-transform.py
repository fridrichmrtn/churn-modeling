# Databricks notebook source
#
##
### DEFINE PARAMS

dbutils.widgets.text("branch", "rees46")
branch = dbutils.widgets.get("branch")
params = {"rees46":{"func":rees46_get_events, "dataInPath":"dbfs:/mnt/rees46/raw", "dataOutPath":"dbfs:/mnt/rees46/delta/events"}}

### WRAPPER
def data_preprocess(params):
    get_events = params["func"]
    events = get_events(params["dataInPath"])
    save_events(events, params["dataOutPath"])
#
##
### WORKHORSE HERE

data_preprocess(params[branch])
