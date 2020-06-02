package api

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/timescale/timescale-prometheus/pkg/util"
)

type Metrics struct {
	LeaderGauge         prometheus.Gauge
	ReceivedSamples     prometheus.Counter
	FailedSamples       prometheus.Counter
	SentSamples         prometheus.Counter
	SentBatchDuration   prometheus.Histogram
	WriteThroughput     *util.ThroughputCalc
	LastRequestUnixNano int64
}