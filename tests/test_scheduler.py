"""Tests for task scheduler."""

import pytest
import asyncio
from datetime import datetime, timezone

from src.scheduler.scheduler import (
    Job,
    JobRun,
    JobStatus,
    Scheduler,
)


class TestJobRun:
    """Tests for JobRun dataclass."""

    def test_duration_while_running(self):
        """Duration should calculate from start to now when running."""
        run = JobRun(started_at=datetime.now(timezone.utc))

        # Should be a small positive number
        assert run.duration_seconds >= 0
        assert run.duration_seconds < 1

    def test_duration_after_completion(self):
        """Duration should calculate from start to end when complete."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        run = JobRun(started_at=start, ended_at=end)

        assert run.duration_seconds == 5.0


class TestJob:
    """Tests for Job dataclass."""

    def test_next_run_when_never_run(self):
        """Next run should be now if never run before."""
        async def dummy():
            pass

        job = Job(name="test", func=dummy, interval_seconds=60)

        # Should be approximately now
        assert job.next_run is not None
        diff = (job.next_run - datetime.now(timezone.utc)).total_seconds()
        assert abs(diff) < 1

    def test_next_run_disabled(self):
        """Next run should be None if disabled."""
        async def dummy():
            pass

        job = Job(name="test", func=dummy, interval_seconds=60, enabled=False)

        assert job.next_run is None

    def test_success_rate_no_history(self):
        """Success rate should be 1.0 with no history."""
        async def dummy():
            pass

        job = Job(name="test", func=dummy, interval_seconds=60)

        assert job.success_rate == 1.0

    def test_success_rate_with_history(self):
        """Success rate should calculate from history."""
        async def dummy():
            pass

        job = Job(name="test", func=dummy, interval_seconds=60)

        # Add 3 successes and 1 failure
        job.history = [
            JobRun(
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                status=JobStatus.COMPLETED,
            ),
            JobRun(
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                status=JobStatus.COMPLETED,
            ),
            JobRun(
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                status=JobStatus.COMPLETED,
            ),
            JobRun(
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                status=JobStatus.FAILED,
            ),
        ]

        assert job.success_rate == 0.75

    def test_to_dict(self):
        """Job should convert to dictionary."""
        async def dummy():
            pass

        job = Job(name="test", func=dummy, interval_seconds=60)

        d = job.to_dict()

        assert d["name"] == "test"
        assert d["interval_seconds"] == 60
        assert d["enabled"] is True
        assert "success_rate" in d


class TestScheduler:
    """Tests for Scheduler."""

    def test_add_job(self):
        """Should add job to scheduler."""
        scheduler = Scheduler()

        async def dummy():
            pass

        job = scheduler.add_job("test", dummy, interval_seconds=60)

        assert job.name == "test"
        assert scheduler.get_job("test") is not None

    def test_add_duplicate_job_raises(self):
        """Adding duplicate job name should raise."""
        scheduler = Scheduler()

        async def dummy():
            pass

        scheduler.add_job("test", dummy, interval_seconds=60)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_job("test", dummy, interval_seconds=60)

    def test_remove_job(self):
        """Should remove job from scheduler."""
        scheduler = Scheduler()

        async def dummy():
            pass

        scheduler.add_job("test", dummy, interval_seconds=60)
        scheduler.remove_job("test")

        assert scheduler.get_job("test") is None

    def test_enable_disable_job(self):
        """Should enable/disable jobs."""
        scheduler = Scheduler()

        async def dummy():
            pass

        scheduler.add_job("test", dummy, interval_seconds=60, enabled=False)

        assert scheduler.get_job("test").enabled is False

        scheduler.enable_job("test")
        assert scheduler.get_job("test").enabled is True

        scheduler.disable_job("test")
        assert scheduler.get_job("test").enabled is False

    def test_get_all_jobs(self):
        """Should return all jobs."""
        scheduler = Scheduler()

        async def dummy():
            pass

        scheduler.add_job("job1", dummy, interval_seconds=60)
        scheduler.add_job("job2", dummy, interval_seconds=120)

        jobs = scheduler.get_all_jobs()

        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_run_job_once(self):
        """Should run job immediately."""
        scheduler = Scheduler()
        counter = {"value": 0}

        async def increment():
            counter["value"] += 1

        scheduler.add_job("counter", increment, interval_seconds=60)

        result = await scheduler.run_job_once("counter")

        assert result is True
        assert counter["value"] == 1

    @pytest.mark.asyncio
    async def test_run_job_once_not_found(self):
        """Should return False for unknown job."""
        scheduler = Scheduler()

        result = await scheduler.run_job_once("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_job_execution_success(self):
        """Job should complete successfully."""
        scheduler = Scheduler()
        executed = {"value": False}

        async def job_func():
            executed["value"] = True

        scheduler.add_job("test", job_func, interval_seconds=60)

        result = await scheduler._execute_job(scheduler.get_job("test"))

        assert result is True
        assert executed["value"] is True

        job = scheduler.get_job("test")
        assert job.last_run is not None
        assert job.last_run.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_job_execution_failure(self):
        """Job should handle failures with retries."""
        scheduler = Scheduler()

        async def failing_job():
            raise ValueError("Test error")

        job = scheduler.add_job(
            "failing",
            failing_job,
            interval_seconds=60,
            max_retries=1,
        )
        job.retry_delay_seconds = 0.01  # Fast retries for test

        result = await scheduler._execute_job(job)

        assert result is False
        assert job.last_run.status == JobStatus.FAILED
        assert job._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_error_callback(self):
        """Should call error callback on failure."""
        errors = []

        async def error_handler(job_name: str, error: Exception):
            errors.append((job_name, str(error)))

        scheduler = Scheduler(error_callback=error_handler)

        async def failing_job():
            raise ValueError("Test error")

        job = scheduler.add_job(
            "failing",
            failing_job,
            interval_seconds=60,
            max_retries=0,
        )

        await scheduler._execute_job(job)

        assert len(errors) == 1
        assert errors[0][0] == "failing"
        assert "Test error" in errors[0][1]

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Should start and stop cleanly."""
        scheduler = Scheduler()
        counter = {"value": 0}

        async def increment():
            counter["value"] += 1

        scheduler.add_job(
            "counter",
            increment,
            interval_seconds=0.1,
            run_immediately=True,
        )

        await scheduler.start()

        # Wait for at least one execution
        await asyncio.sleep(0.15)

        await scheduler.stop()

        assert counter["value"] >= 1

    @pytest.mark.asyncio
    async def test_run_immediately(self):
        """Job with run_immediately should execute on start."""
        scheduler = Scheduler()
        executed = {"value": False}

        async def job_func():
            executed["value"] = True

        scheduler.add_job(
            "test",
            job_func,
            interval_seconds=100,  # Long interval
            run_immediately=True,
        )

        await scheduler.start()
        await asyncio.sleep(0.1)
        await scheduler.stop()

        assert executed["value"] is True

    def test_get_status(self):
        """Should return scheduler status."""
        scheduler = Scheduler()

        async def dummy():
            pass

        scheduler.add_job("job1", dummy, interval_seconds=60)
        scheduler.add_job("job2", dummy, interval_seconds=60, enabled=False)

        status = scheduler.get_status()

        assert status["running"] is False
        assert status["total_jobs"] == 2
        assert status["enabled_jobs"] == 1
        assert len(status["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self):
        """Should limit job history size."""
        scheduler = Scheduler(max_history_per_job=3)
        counter = {"value": 0}

        async def increment():
            counter["value"] += 1

        scheduler.add_job("counter", increment, interval_seconds=60)
        job = scheduler.get_job("counter")

        # Run job multiple times
        for _ in range(5):
            await scheduler._execute_job(job)

        assert len(job.history) == 3
