// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.spark.sql.connect.nativebridge.engine;

import java.io.Serializable;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.arrow.ArrowConverters$;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;
import scala.collection.AbstractIterator;
import scala.collection.Iterator;
import scala.runtime.AbstractFunction1;

/** Only conversion settings are serialized; Spark owns Arrow allocation and task cleanup. */
final class ArrowPartitionConverter
    extends AbstractFunction1<Iterator<InternalRow>, Iterator<Tuple2<byte[], Long>>>
    implements Serializable {
  private static final long serialVersionUID = 1L;
  private static final long MAX_ESTIMATED_BATCH_BYTES = 2 * 1024 * 1024;
  private final StructType schema;
  private final String timeZone;
  private final boolean largeVarTypes;
  private final int maxRows;

  ArrowPartitionConverter(StructType schema, String timeZone, boolean largeVarTypes, int maxRows) {
    this.schema = schema;
    this.timeZone = timeZone;
    this.largeVarTypes = largeVarTypes;
    this.maxRows = maxRows;
  }

  @Override
  public Iterator<Tuple2<byte[], Long>> apply(Iterator<InternalRow> rows) {
    try {
      var batches =
          ArrowConverters$.MODULE$.toBatchWithSchemaIterator(rows, schema, maxRows,
              MAX_ESTIMATED_BATCH_BYTES, timeZone, false, largeVarTypes);
      return new AbstractIterator<Tuple2<byte[], Long>>() {
        @Override
        public boolean hasNext() {
          try {
            return batches.hasNext();
          } catch (Exception error) {
            throw SparkJvmEngine.executorConversionFailure(error);
          }
        }
        @Override
        public Tuple2<byte[], Long> next() {
          try {
            byte[] data = SparkJvmEngine.checkedIpc(batches.next());
            return new Tuple2<>(data, batches.rowCountInLastBatch());
          } catch (Exception error) {
            throw SparkJvmEngine.executorConversionFailure(error);
          }
        }
      };
    } catch (Exception error) {
      throw SparkJvmEngine.executorConversionFailure(error);
    }
  }
}
