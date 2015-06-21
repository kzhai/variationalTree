#!/bin/bash

CorpusName=tree-synthetic
TreeName=tree0

NumTopic=5
NumIteration=50
NumMapper=10
NumReducer=$NumTopic

SnapshotInterval=10

nohup bash /fs/clip-lsbi/Workspace/variational/src/vb/prior/tree/dumbo/launch.sh \
    $CorpusName \
    $TreeName \
    $NumTopic \
    $NumIteration \
    $NumMapper \
    $NumReducer \
    $SnapshotInterval \
    > /fs/clip-lsbi/Workspace/variational/output/$CorpusName/$TreeName-K$NumTopic-I$NumIteration-M$NumMapper-R$NumReducer.output &