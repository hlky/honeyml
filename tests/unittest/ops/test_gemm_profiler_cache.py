#  Copyright 2025 hlky. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import tempfile
import unittest
from unittest.mock import patch

from honey.backend.profiler_cache import ProfileCacheDB

from honey.compiler import compile_model, ops
from honey.frontend import IntImm, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import env_variables


_LOGGER = logging.getLogger(__name__)


class GemmProfilerCacheTestCase(unittest.TestCase):
    def _test(
        self,
        first_dim,
        logger,
        test_name="gemm_rcr",
        k=128,
        n=8,
    ):
        target = detect_target()

        X = Tensor(
            shape=[first_dim, k],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[n, k],
            dtype="float16",
            name="input_1",
            is_input=True,
        )
        OP = ops.gemm_rcr()
        Y = OP(X, W)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        with self.assertLogs(
            logger=logger,
            level="INFO",
        ) as logs:
            compile_model(
                Y,
                target,
                "./tmp",
                test_name,
            )

        return "\n".join(logs.output)

    def _run_test(
        self,
        first_dim,
        test_name,
        logger,
        cache_dir,
    ):
        with env_variables(
            TRICK_CI_ENV="1",
            CACHE_DIR=f"{cache_dir}/{test_name}",
        ):
            return self._test(
                first_dim=first_dim,
                logger=logger,
                test_name=test_name,
            )

    def test_gemm_profiler_cache(self):
        first_dim = IntImm(4)
        test_name = "gemm_rcr_profiler_cache"
        logger = "honey.compiler.transform.profile"

        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            run1_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 1 profilers", run1_logs)

            run2_logs = self._run_test(
                first_dim=first_dim,
                test_name=test_name,
                logger=logger,
                cache_dir=tmp_dirname,
            )
            self.assertIn("generated 0 profilers", run2_logs)

    def test_gemm_profiler_cache_versioning(self):
        first_dim = IntImm(4)
        test_name = "gemm_rcr_profiler_cache_versioning"
        logger = "honey.backend.profiler_cache"
        cache_version_property = "gemm_cache_version"
        target_name = detect_target().name()

        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=1,  # version
            ):
                run1_before_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_gemm_1' does not exist in the db",
                    run1_before_version_change_logs,
                )

                run2_before_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_gemm_1' exists in the db",
                    run2_before_version_change_logs,
                )

            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=2,  # version
            ):
                run1_after_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_gemm_2' does not exist in the db",
                    run1_after_version_change_logs,
                )

                run2_after_version_change_logs = self._run_test(
                    first_dim=first_dim,
                    test_name=test_name,
                    logger=logger,
                    cache_dir=tmp_dirname,
                )
                self.assertIn(
                    f"table_name='{target_name}_gemm_2' exists in the db",
                    run2_after_version_change_logs,
                )

    def test_gemm_profiler_force_cache(self):
        first_dim = IntImm(4)
        test_name = "gemm_rcr_profiler_force_cache"
        cache_version_property = "gemm_cache_version"

        logger = "honey.backend.profiler_cache"
        _LOGGER.info(f"running {test_name=}")
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _LOGGER.info(f"{tmp_dirname=}")
            with patch.object(
                target=ProfileCacheDB,
                attribute=cache_version_property,
                new=1,  # version
            ):
                _LOGGER.info("force cache with no cache 1")
                with env_variables(Honey_FORCE_PROFILER_CACHE="1"):
                    with self.assertRaisesRegex(
                        RuntimeError, "force_cache is enabled but we could not find"
                    ):
                        self._run_test(
                            first_dim=first_dim,
                            test_name=test_name,
                            logger=logger,
                            cache_dir=tmp_dirname,
                        )

                _LOGGER.info("make cache 1")
                with env_variables(Honey_FORCE_PROFILER_CACHE=None):
                    self._run_test(
                        first_dim=first_dim,
                        test_name=test_name,
                        logger=logger,
                        cache_dir=tmp_dirname,
                    )

                _LOGGER.info("force cache with no cache 1")
                with env_variables(Honey_FORCE_PROFILER_CACHE="1"):
                    self._run_test(
                        first_dim=first_dim,
                        test_name=test_name,
                        logger=logger,
                        cache_dir=tmp_dirname,
                    )


if __name__ == "__main__":
    unittest.main()
