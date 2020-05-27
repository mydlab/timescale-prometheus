// This file and its contents are licensed under the Apache License 2.0.
// Please see the included NOTICE for copyright information and
// LICENSE for a copy of the license.

package pgmodel

import (
	"bytes"
	"database/sql"
	"fmt"
	"io"
	"io/ioutil"
	"strings"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source"
	"github.com/golang-migrate/migrate/v4/source/httpfs"
	"github.com/timescale/timescale-prometheus/pkg/log"
	"github.com/timescale/timescale-prometheus/pkg/pgmodel/migrations"
)

const (
	timescaleInstall            = "CREATE EXTENSION IF NOT EXISTS timescaledb WITH SCHEMA public;"
	extensionInstall            = "CREATE EXTENSION IF NOT EXISTS timescale_prometheus_extra WITH SCHEMA %s;"
	metadataUpdateWithExtension = "SELECT update_tsprom_metadata($1, $2, $3)"
	metadataUpdateNoExtension   = "INSERT INTO _timescaledb_catalog.metadata(key, value, include_in_telemetry) VALUES ($1, $2, $3) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, include_in_telemetry = EXCLUDED.include_in_telemetry"
)

type mySrc struct {
	source.Driver
}

type VersionInfo struct {
	Version    string
	CommitHash string
}

func (t *mySrc) replaceSchemaNames(r io.ReadCloser) (io.ReadCloser, error) {
	buf := new(bytes.Buffer)
	_, err := buf.ReadFrom(r)
	if err != nil {
		return r, err
	}
	err = r.Close()
	if err != nil {
		return r, err
	}
	s := buf.String()
	s = strings.ReplaceAll(s, "SCHEMA_CATALOG", catalogSchema)
	s = strings.ReplaceAll(s, "SCHEMA_EXT", extSchema)
	s = strings.ReplaceAll(s, "SCHEMA_PROM", promSchema)
	s = strings.ReplaceAll(s, "SCHEMA_SERIES", seriesViewSchema)
	s = strings.ReplaceAll(s, "SCHEMA_METRIC", metricViewSchema)
	s = strings.ReplaceAll(s, "SCHEMA_DATA", dataSchema)
	s = strings.ReplaceAll(s, "SCHEMA_DATA_SERIES", dataSeriesSchema)
	s = strings.ReplaceAll(s, "SCHEMA_INFO", infoSchema)
	r = ioutil.NopCloser(strings.NewReader(s))
	return r, err
}

func (t *mySrc) ReadUp(version uint) (r io.ReadCloser, identifier string, err error) {
	r, identifier, err = t.Driver.ReadUp(version)
	if err != nil {
		return
	}
	r, err = t.replaceSchemaNames(r)
	return
}

func (t *mySrc) ReadDown(version uint) (r io.ReadCloser, identifier string, err error) {
	r, identifier, err = t.Driver.ReadDown(version)
	if err != nil {
		return
	}
	r, err = t.replaceSchemaNames(r)
	return
}

func metadataUpdate(db *sql.DB, withExtension bool, key string, value string) {
	/* Ignore error if it doesn't work */
	if withExtension {
		_, _ = db.Exec(metadataUpdateWithExtension, key, value, true)
	} else {
		_, _ = db.Exec(metadataUpdateNoExtension, key, value, true)
	}
}

// Migrate performs a database migration to the latest version
func Migrate(db *sql.DB, versionInfo VersionInfo) (err error) {
	// The migration table will be put in the public schema not in any of our schema because we never want to drop it and
	// our scripts and our last down script drops our shemas
	driver, err := postgres.WithInstance(db, &postgres.Config{MigrationsTable: "prom_schema_migrations"})
	if err != nil {
		return fmt.Errorf("cannot create driver due to %w", err)
	}

	_, err = db.Exec(timescaleInstall)
	if err != nil {
		return fmt.Errorf("timescaledb failed to install due to %w", err)
	}

	src, err := httpfs.New(migrations.SqlFiles, "/")
	if err != nil {
		return err
	}
	src = &mySrc{src}

	m, err := migrate.NewWithInstance("SqlFiles", src, "Postgresql", driver)
	if err != nil {
		return err
	}
	defer func() {
		sourceErr, databaseErr := m.Close()
		//don't override error if already set
		if err != nil {
			return
		}
		if sourceErr != nil {
			err = sourceErr
			return
		}
		if databaseErr != nil {
			err = databaseErr
			return
		}
	}()

	err = m.Up()
	//ignore no change errors as we want this idempotent. Being up to date is not a bad thing.
	if err == migrate.ErrNoChange {
		err = nil
	}
	if err != nil {
		return err
	}

	_, extErr := db.Exec(fmt.Sprintf(extensionInstall, extSchema))
	if extErr != nil {
		log.Warn("msg", "timescale_prometheus_extra extension not installed", "cause", extErr)
	}

	// Insert metadata.
	metadataUpdate(db, extErr == nil, "version", versionInfo.Version)
	metadataUpdate(db, extErr == nil, "commit_hash", versionInfo.CommitHash)

	return nil
}
