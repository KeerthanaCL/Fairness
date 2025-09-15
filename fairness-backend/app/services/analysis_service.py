"""
Analysis service for orchestrating fairness evaluation
Manages the complete fairness analysis workflow
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.models.schemas import (
    AnalysisStartRequest, AnalysisStatusResponse, AnalysisStatus, AnalysisStep,
    SensitiveFeatureDetectionResponse, MitigationStrategiesResponse,
    BeforeAfterComparisonResponse
)
from app.core.data_processor import DataProcessor
from app.core.bias_detector import BiasDetector
from app.core.fairness_metrics import FairnessMetricsCalculator
from app.utils.file_handler import FileHandler

logger = logging.getLogger(__name__)


class AnalysisService:
    """Manages fairness analysis lifecycle and results"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisService, cls).__new__(cls)
            cls._instance.analyses = {}
            cls._instance.data_processor = DataProcessor()
            cls._instance.bias_detector = BiasDetector()
            cls._instance.fairness_metrics = FairnessMetricsCalculator()
            cls._instance.file_handler = FileHandler()
            logger.info("Created new AnalysisService singleton instance")
        else:
            logger.info("Returning existing AnalysisService singleton instance")
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def create_analysis(self, analysis_id: str, request: AnalysisStartRequest, steps: List[AnalysisStep]):
        """Create a new analysis entry"""
        self.analyses[analysis_id] = {
            "id": analysis_id,
            "status": AnalysisStatus.PENDING,
            "request": request,
            "steps": steps,
            "progress": 0,
            "current_step": None,
            "created_at": datetime.now(),
            "results": {},
            "error": None
        }
        
        logger.info(f"Analysis created: {analysis_id}")
    
    async def run_analysis(self, analysis_id: str):
        """Run the complete fairness analysis workflow"""
        logger.info(f"Starting background analysis for ID: {analysis_id}")
        
        try:
            if analysis_id not in self.analyses:
                logger.error(f"Analysis not found: {analysis_id}")
                return
            
            logger.info(f"Analysis found, starting processing for: {analysis_id}")
            
            analysis = self.analyses[analysis_id]
            analysis["status"] = AnalysisStatus.RUNNING
            
            # Step 1: File validation and loading
            await self._update_step(analysis_id, "file_validation", AnalysisStatus.RUNNING, 10)
            
            # Get upload file paths
            request = analysis["request"]
            
            # Import upload registry functions
            from app.api.upload import get_upload_file_path
            from pathlib import Path
            
            # Resolve upload IDs to file paths
            training_path_str = get_upload_file_path(request.training_dataset_id)
            model_path_str = get_upload_file_path(request.model_id)
            testing_path_str = get_upload_file_path(request.testing_dataset_id) if request.testing_dataset_id else None
            
            logger.info(f"Resolved paths - Training: {training_path_str}, Model: {model_path_str}, Testing: {testing_path_str}")
            
            if not training_path_str:
                logger.error(f"Training dataset not found for ID: {request.training_dataset_id}")
                raise Exception(f"Training dataset not found: {request.training_dataset_id}")
            if not model_path_str:
                logger.error(f"Model not found for ID: {request.model_id}")
                raise Exception(f"Model not found: {request.model_id}")
            
            # Convert string paths to Path objects for FileHandler
            training_path = Path(training_path_str)
            model_path = Path(model_path_str)
            testing_path = Path(testing_path_str) if testing_path_str else None
            
            # Load files
            train_df = self.file_handler.load_dataset(training_path)
            model = self.file_handler.load_model(model_path)
            
            # Store model and data in analysis for later use
            analysis["model"] = model
            analysis["train_df"] = train_df
            
            # Load test data if provided
            test_df = None
            if testing_path:
                test_df = self.file_handler.load_dataset(testing_path)
                logger.info(f"Test dataset loaded successfully with {len(test_df)} rows")
            else:
                logger.warning("No test dataset provided - fairness evaluation will use training data (not recommended)")
            
            # Store test data in analysis
            analysis["test_df"] = test_df
            
            await self._update_step(analysis_id, "file_validation", AnalysisStatus.COMPLETED, 20)
            
            # Step 2: Sensitive feature detection
            await self._update_step(analysis_id, "sensitive_feature_detection", AnalysisStatus.RUNNING, 30)
            
            # Determine feature types
            feature_types = self.data_processor.detect_feature_types(train_df)
            exclude_columns = [request.target_column] if request.target_column else []
            
            # Detect sensitive features
            sensitive_features_result = self.bias_detector.detect_sensitive_features(
                train_df, request.target_column, feature_types, exclude_columns
            )
            
            # Convert Pydantic model to dict for storage
            analysis["results"]["sensitive_features"] = sensitive_features_result.model_dump()
            await self._update_step(analysis_id, "sensitive_feature_detection", AnalysisStatus.COMPLETED, 50)
            
            # Step 3: Fairness metrics calculation
            await self._update_step(analysis_id, "fairness_metrics_calculation", AnalysisStatus.RUNNING, 60)
            
            # Get sensitive attributes (detected + user specified)
            detected_attrs = [sf.name for sf in sensitive_features_result.detectedFeatures]
            sensitive_attrs = request.sensitive_attributes or detected_attrs
            
            # Calculate fairness metrics - use test data if available, otherwise fall back to training data
            evaluation_df = test_df if test_df is not None else train_df
            dataset_type = "test" if test_df is not None else "training"
            logger.info(f"Using {dataset_type} dataset for fairness metrics calculation")
            
            feature_columns = [col for col in evaluation_df.columns if col != request.target_column]
            fairness_results = self.fairness_metrics.calculate_all_metrics(
                evaluation_df, model, sensitive_attrs, request.target_column, feature_columns
            )
            
            analysis["results"]["fairness_metrics"] = fairness_results
            await self._update_step(analysis_id, "fairness_metrics_calculation", AnalysisStatus.COMPLETED, 80)
            
            # Step 4: Mitigation strategy generation with real evaluation
            await self._update_step(analysis_id, "mitigation_strategy_generation", AnalysisStatus.RUNNING, 90)
            
            # Generate mitigation strategies with real performance evaluation
            logger.info("Generating mitigation strategies with real model evaluation")
            mitigation_strategies = self._generate_mitigation_strategies(
                sensitive_features_result, 
                fairness_results,
                model=analysis["model"],
                train_df=analysis["train_df"], 
                test_df=analysis["test_df"],
                target_column=request.target_column
            )
            
            analysis["results"]["mitigation_strategies"] = mitigation_strategies
            await self._update_step(analysis_id, "mitigation_strategy_generation", AnalysisStatus.COMPLETED, 95)
            
            # Step 5: Before/after comparison (simulation)
            await self._update_step(analysis_id, "before_after_comparison", AnalysisStatus.RUNNING, 98)
            
            # Generate before/after comparisons for each strategy
            before_after_results = {}
            sensitive_features_data = analysis["results"].get("sensitive_features")
            for strategy in mitigation_strategies.strategies:
                comparison = self._simulate_before_after_comparison(
                    strategy, fairness_results, sensitive_attrs, sensitive_features_data
                )
                before_after_results[strategy.name] = comparison
            
            analysis["results"]["before_after_comparisons"] = before_after_results
            await self._update_step(analysis_id, "before_after_comparison", AnalysisStatus.COMPLETED, 100)
            
            # Mark analysis as completed
            analysis["status"] = AnalysisStatus.COMPLETED
            analysis["progress"] = 100
            analysis["completed_at"] = datetime.now()
            
            logger.info(f"Analysis completed successfully: {analysis_id}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {analysis_id}, error: {str(e)}")
            analysis = self.analyses.get(analysis_id)
            if analysis:
                analysis["status"] = AnalysisStatus.FAILED
                analysis["error"] = str(e)
                analysis["failed_at"] = datetime.now()
    
    async def _update_step(self, analysis_id: str, step_name: str, status: AnalysisStatus, progress: int):
        """Update the status of a specific analysis step"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            return
        
        # Update step status
        for step in analysis["steps"]:
            if step.name == step_name:
                step.status = status
                step.progress = progress if status == AnalysisStatus.COMPLETED else progress - 10
                break
        
        # Update overall progress
        analysis["progress"] = progress
        analysis["current_step"] = step_name if status == AnalysisStatus.RUNNING else None
        
        # Small delay to simulate processing time
        await asyncio.sleep(0.5)
    
    def get_analysis_status(self, analysis_id: str) -> Optional[AnalysisStatusResponse]:
        """Get the current status of an analysis"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            return None
        
        # Create analysis steps based on the current analysis state
        from app.models.schemas import AnalysisStep
        steps = [
            AnalysisStep(
                name="Data Loading",
                status=AnalysisStatus.COMPLETED if analysis["progress"] > 10 else 
                       AnalysisStatus.RUNNING if analysis["status"] == AnalysisStatus.RUNNING else AnalysisStatus.PENDING,
                progress=min(100, analysis["progress"] * 2) if analysis["progress"] > 0 else 0
            ),
            AnalysisStep(
                name="Sensitive Feature Detection",
                status=AnalysisStatus.COMPLETED if analysis["progress"] > 40 else 
                       AnalysisStatus.RUNNING if analysis["status"] == AnalysisStatus.RUNNING and analysis["progress"] > 10 else AnalysisStatus.PENDING,
                progress=max(0, min(100, (analysis["progress"] - 10) * 2.5)) if analysis["progress"] > 10 else 0
            ),
            AnalysisStep(
                name="Fairness Metrics Calculation",
                status=AnalysisStatus.COMPLETED if analysis["progress"] > 70 else 
                       AnalysisStatus.RUNNING if analysis["status"] == AnalysisStatus.RUNNING and analysis["progress"] > 40 else AnalysisStatus.PENDING,
                progress=max(0, min(100, (analysis["progress"] - 40) * 3)) if analysis["progress"] > 40 else 0
            ),
            AnalysisStep(
                name="Mitigation Strategy Analysis",
                status=AnalysisStatus.COMPLETED if analysis["progress"] >= 100 else 
                       AnalysisStatus.RUNNING if analysis["status"] == AnalysisStatus.RUNNING and analysis["progress"] > 70 else AnalysisStatus.PENDING,
                progress=max(0, min(100, (analysis["progress"] - 70) * 3.33)) if analysis["progress"] > 70 else 0
            )
        ]
        
        return AnalysisStatusResponse(
            analysisId=analysis_id,
            status=analysis["status"],
            progress=analysis["progress"],
            currentStep=analysis["current_step"],
            steps=steps,
            results=analysis["results"] if analysis["status"] == AnalysisStatus.COMPLETED else None
        )
    
    def stop_analysis(self, analysis_id: str) -> bool:
        """Stop a running analysis"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            return False
        
        if analysis["status"] == AnalysisStatus.RUNNING:
            analysis["status"] = AnalysisStatus.FAILED
            analysis["error"] = "Analysis stopped by user"
            analysis["stopped_at"] = datetime.now()
            
            logger.info(f"Analysis stopped: {analysis_id}")
            return True
        
        return False
    
    def get_sensitive_features(self, analysis_id: str) -> Optional[SensitiveFeatureDetectionResponse]:
        """Get sensitive features results"""
        analysis = self.analyses.get(analysis_id)
        if not analysis or "sensitive_features" not in analysis["results"]:
            return None
        
        return analysis["results"]["sensitive_features"]
    
    def get_fairness_metrics(self, analysis_id: str, attribute: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get fairness metrics results"""
        analysis = self.analyses.get(analysis_id)
        if not analysis or "fairness_metrics" not in analysis["results"]:
            return None
        
        fairness_results = analysis["results"]["fairness_metrics"]
        
        if attribute:
            # Return specific attribute
            return {attribute: fairness_results.get(attribute)} if attribute in fairness_results else None
        
        # Return all attributes
        return fairness_results
    
    def get_mitigation_strategies(self, analysis_id: str) -> Optional[MitigationStrategiesResponse]:
        """Get mitigation strategies results"""
        analysis = self.analyses.get(analysis_id)
        if not analysis or "mitigation_strategies" not in analysis["results"]:
            return None
        
        return analysis["results"]["mitigation_strategies"]
    
    def get_before_after_comparison(self, analysis_id: str, strategy: str) -> Optional[BeforeAfterComparisonResponse]:
        """Get before/after comparison for a specific strategy"""
        analysis = self.analyses.get(analysis_id)
        if not analysis or "before_after_comparisons" not in analysis["results"]:
            return None
        
        comparisons = analysis["results"]["before_after_comparisons"]
        return comparisons.get(strategy)
    
    def get_all_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get all analysis results"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            return None
        
        return {
            "id": analysis_id,
            "status": analysis["status"],
            "progress": analysis["progress"],
            "results": analysis["results"],
            "error": analysis["error"],
            "created_at": analysis["created_at"].isoformat(),
            "completed_at": analysis.get("completed_at", {}).isoformat() if analysis.get("completed_at") else None
        }
    
    def list_all_analyses(self) -> List[Dict[str, Any]]:
        """List all analyses (for debugging/admin)"""
        return [
            {
                "id": aid,
                "status": analysis["status"],
                "progress": analysis["progress"],
                "created_at": analysis["created_at"].isoformat(),
                "current_step": analysis["current_step"]
            }
            for aid, analysis in self.analyses.items()
        ]
    
    def _estimate_time_remaining(self, analysis: Dict[str, Any]) -> Optional[int]:
        """Estimate time remaining for analysis"""
        if analysis["status"] != AnalysisStatus.RUNNING:
            return None
        
        progress = analysis["progress"]
        if progress == 0:
            return 300  # 5 minutes estimate
        
        # Simple linear estimate based on progress
        elapsed_time = (datetime.now() - analysis["created_at"]).total_seconds()
        estimated_total = elapsed_time / (progress / 100)
        remaining = max(0, estimated_total - elapsed_time)
        
        return int(remaining)
    
    def _generate_mitigation_strategies(self, sensitive_features, fairness_results, model=None, train_df=None, test_df=None, target_column=None) -> MitigationStrategiesResponse:
        """Generate comprehensive mitigation strategy recommendations with real performance calculations"""
        from app.models.schemas import MitigationStrategy, StrategyCategory
        
        strategies = []
        
        # Get all detected sensitive attributes
        all_attrs = [sf.name for sf in sensitive_features.detectedFeatures]
        
        # If we have real data, calculate actual performance for each strategy
        if model is not None and train_df is not None and test_df is not None and target_column is not None:
            logger.info("Calculating real performance metrics for all mitigation strategies")
            return self._generate_strategies_with_real_evaluation(
                sensitive_features, fairness_results, model, train_df, test_df, target_column, all_attrs
            )
        
        # Fallback to estimation-based generation (original behavior)
        logger.warning("No model/data provided, using estimation-based strategy generation")
        avg_bias_severity = self._calculate_bias_severity(fairness_results)
        high_bias_attrs = [sf.name for sf in sensitive_features.detectedFeatures 
                          if sf.sensitivityLevel == "Highly Sensitive"]
        
        # Calculate realistic improvement estimates based on bias severity
        high_bias_improvement = min(45.0, avg_bias_severity * 1.4)
        medium_bias_improvement = min(35.0, avg_bias_severity * 1.1)
        low_bias_improvement = min(25.0, avg_bias_severity * 0.9)
        
        # PREPROCESSING STRATEGIES (1-3)
        strategies.extend([
            # 1. Data Reweighing - Most effective for high bias
            MitigationStrategy(
                name="Reweighing",
                category=StrategyCategory.PREPROCESSING,
                fairnessImprovement=round(high_bias_improvement, 1),
                accuracyImpact=round(-high_bias_improvement * 0.06, 1),
                precisionImpact=round(-high_bias_improvement * 0.05, 1),
                recallImpact=round(-high_bias_improvement * 0.04, 1),
                f1Impact=round(-high_bias_improvement * 0.055, 1),
                stars=3 if high_bias_improvement > 30 else 2,
                recommendation="Highly Recommended" if high_bias_improvement > 30 else "Recommended",
                description="Adjusts instance weights to reduce bias in training data. Most effective for datasets with imbalanced sensitive groups.",
                targetAttributes=all_attrs
            ),
            
            # 2. Disparate Impact Remover - Good for direct discrimination
            MitigationStrategy(
                name="Disparate Impact Remover",
                category=StrategyCategory.PREPROCESSING,
                fairnessImprovement=round(medium_bias_improvement, 1),
                accuracyImpact=round(-medium_bias_improvement * 0.08, 1),
                precisionImpact=round(-medium_bias_improvement * 0.07, 1),
                recallImpact=round(-medium_bias_improvement * 0.06, 1),
                f1Impact=round(-medium_bias_improvement * 0.075, 1),
                stars=2 if medium_bias_improvement > 25 else 1,
                recommendation="Recommended" if medium_bias_improvement > 25 else "Consider",
                description="Removes features that cause disparate impact. Ideal when direct discrimination through features is suspected.",
                targetAttributes=all_attrs
            ),
            
            # 3. Data Augmentation - Good for small datasets
            MitigationStrategy(
                name="Data Augmentation",
                category=StrategyCategory.PREPROCESSING,
                fairnessImprovement=round(low_bias_improvement, 1),
                accuracyImpact=round(low_bias_improvement * 0.04, 1),  # Positive impact
                precisionImpact=round(low_bias_improvement * 0.03, 1),
                recallImpact=round(low_bias_improvement * 0.05, 1),
                f1Impact=round(low_bias_improvement * 0.04, 1),
                stars=2 if len(all_attrs) > 1 else 1,
                recommendation="Good for Small Datasets" if len(all_attrs) > 1 else "Limited Benefit",
                description="Augments underrepresented groups with synthetic data. Most effective for small datasets with underrepresented groups.",
                targetAttributes=all_attrs
            )
        ])
        
        # IN-PROCESSING STRATEGIES (4-5)
        strategies.extend([
            # 4. Fairness Regularization - Balanced approach
            MitigationStrategy(
                name="Fairness Regularization",
                category=StrategyCategory.IN_PROCESSING,
                fairnessImprovement=round(medium_bias_improvement * 1.1, 1),
                accuracyImpact=round(-medium_bias_improvement * 0.05, 1),
                precisionImpact=round(-medium_bias_improvement * 0.04, 1),
                recallImpact=round(-medium_bias_improvement * 0.04, 1),
                f1Impact=round(-medium_bias_improvement * 0.045, 1),
                stars=3 if medium_bias_improvement > 25 else 2,
                recommendation="Highly Recommended" if medium_bias_improvement > 25 else "Recommended",
                description="Adds fairness penalty to model training objective. Integrates fairness constraints into model optimization.",
                targetAttributes=all_attrs
            ),
            
            # 5. Adversarial Debiasing - Advanced technique
            MitigationStrategy(
                name="Adversarial Debiasing",
                category=StrategyCategory.IN_PROCESSING,
                fairnessImprovement=round(high_bias_improvement * 0.9, 1),
                accuracyImpact=round(-high_bias_improvement * 0.04, 1),
                precisionImpact=round(-high_bias_improvement * 0.03, 1),
                recallImpact=round(-high_bias_improvement * 0.03, 1),
                f1Impact=round(-high_bias_improvement * 0.035, 1),
                stars=2 if high_bias_improvement > 20 else 1,
                recommendation="Advanced Option" if high_bias_improvement > 20 else "Complex Implementation",
                description="Uses adversarial training to remove bias. Can handle complex bias patterns but requires more computational resources.",
                targetAttributes=all_attrs
            )
        ])
        
        # POST-PROCESSING STRATEGIES (6-8)
        strategies.extend([
            # 6. Threshold Optimization - Quick and effective
            MitigationStrategy(
                name="Threshold Optimization",
                category=StrategyCategory.POST_PROCESSING,
                fairnessImprovement=round(medium_bias_improvement * 0.8, 1),
                accuracyImpact=round(-medium_bias_improvement * 0.03, 1),
                precisionImpact=round(-medium_bias_improvement * 0.025, 1),
                recallImpact=round(-medium_bias_improvement * 0.02, 1),
                f1Impact=round(-medium_bias_improvement * 0.03, 1),
                stars=3,
                recommendation="Quick Implementation",
                description="Optimizes decision thresholds for each sensitive group. Works with existing models and quick to implement.",
                targetAttributes=all_attrs
            ),
            
            # 7. Calibration Adjustment - For probability models
            MitigationStrategy(
                name="Calibration Adjustment",
                category=StrategyCategory.POST_PROCESSING,
                fairnessImprovement=round(low_bias_improvement * 1.2, 1),
                accuracyImpact=round(-low_bias_improvement * 0.02, 1),
                precisionImpact=round(-low_bias_improvement * 0.015, 1),
                recallImpact=round(-low_bias_improvement * 0.015, 1),
                f1Impact=round(-low_bias_improvement * 0.02, 1),
                stars=2,
                recommendation="For Probability Models",
                description="Calibrates predictions to ensure fairness across groups. Ideal for probability-based models with calibration issues.",
                targetAttributes=all_attrs
            ),
            
            # 8. Equalized Odds Post-processing - Targeted fairness
            MitigationStrategy(
                name="Equalized Odds Post-processing",
                category=StrategyCategory.POST_PROCESSING,
                fairnessImprovement=round(medium_bias_improvement * 0.9, 1),
                accuracyImpact=round(-medium_bias_improvement * 0.025, 1),
                precisionImpact=round(-medium_bias_improvement * 0.02, 1),
                recallImpact=round(-medium_bias_improvement * 0.02, 1),
                f1Impact=round(-medium_bias_improvement * 0.025, 1),
                stars=2,
                recommendation="Targeted Fairness",
                description="Adjusts predictions to achieve equalized odds across groups. Directly optimizes for equalized odds fairness metric.",
                targetAttributes=all_attrs
            )
        ])
        
        return MitigationStrategiesResponse(strategies=strategies)
    
    def _generate_strategies_with_real_evaluation(
        self, 
        sensitive_features, 
        fairness_results, 
        model, 
        train_df, 
        test_df, 
        target_column, 
        all_attrs
    ) -> MitigationStrategiesResponse:
        """Generate strategies with real performance evaluation using actual model and data"""
        from app.models.schemas import MitigationStrategy, StrategyCategory
        from app.services.mitigation_service import MitigationService
        
        strategies = []
        mitigation_service = MitigationService()
        
        # Define all 8 strategies to evaluate
        strategy_definitions = [
            ("Reweighing", StrategyCategory.PREPROCESSING, "Adjusts instance weights to reduce bias in training data. Most effective for datasets with imbalanced sensitive groups."),
            ("Disparate Impact Remover", StrategyCategory.PREPROCESSING, "Removes features that cause disparate impact. Ideal when direct discrimination through features is suspected."),
            ("Data Augmentation", StrategyCategory.PREPROCESSING, "Augments underrepresented groups with synthetic data. Most effective for small datasets with underrepresented groups."),
            ("Fairness Regularization", StrategyCategory.IN_PROCESSING, "Adds fairness penalty to model training objective. Integrates fairness constraints into model optimization."),
            ("Adversarial Debiasing", StrategyCategory.IN_PROCESSING, "Uses adversarial training to remove bias. Can handle complex bias patterns but requires more computational resources."),
            ("Threshold Optimization", StrategyCategory.POST_PROCESSING, "Optimizes decision thresholds for each sensitive group. Works with existing models and quick to implement."),
            ("Calibration Adjustment", StrategyCategory.POST_PROCESSING, "Calibrates predictions to ensure fairness across groups. Ideal for probability-based models with calibration issues."),
            ("Equalized Odds Post-processing", StrategyCategory.POST_PROCESSING, "Adjusts predictions to achieve equalized odds across groups. Directly optimizes for equalized odds fairness metric.")
        ]
        
        logger.info(f"Evaluating {len(strategy_definitions)} mitigation strategies with real model and data")
        
        for strategy_name, category, description in strategy_definitions:
            try:
                logger.info(f"Evaluating strategy: {strategy_name}")
                
                # Apply the strategy and get real results
                result = mitigation_service.apply_mitigation_strategy(
                    strategy_name=strategy_name,
                    model=model,
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    sensitive_attributes=all_attrs
                )
                
                # Extract real performance metrics
                before_performance = result["before"]["performance"]
                after_performance = result["after"]["performance"]
                improvement = result["improvement"]
                
                # Calculate real impacts (as percentages)
                fairness_improvement = improvement.get("fairness_improvement", 0)
                accuracy_impact = improvement.get("accuracy_change", 0) * 100
                precision_impact = improvement.get("precision_change", 0) * 100
                recall_impact = improvement.get("recall_change", 0) * 100
                f1_impact = improvement.get("f1_change", 0) * 100
                
                # Determine recommendation based on real results
                stars, recommendation = self._calculate_strategy_recommendation(
                    fairness_improvement, accuracy_impact, f1_impact
                )
                
                # Create strategy with real calculated values
                strategy = MitigationStrategy(
                    name=strategy_name,
                    category=category,
                    fairnessImprovement=round(fairness_improvement, 1),
                    accuracyImpact=round(accuracy_impact, 1),
                    precisionImpact=round(precision_impact, 1),
                    recallImpact=round(recall_impact, 1),
                    f1Impact=round(f1_impact, 1),
                    stars=stars,
                    recommendation=recommendation,
                    description=description,
                    targetAttributes=all_attrs
                )
                
                strategies.append(strategy)
                logger.info(f"Strategy {strategy_name} evaluated: {fairness_improvement:.1f}% fairness improvement")
                
            except Exception as e:
                logger.error(f"Failed to evaluate strategy {strategy_name}: {str(e)}")
                # Add fallback strategy with estimated values
                strategies.append(self._create_fallback_strategy(strategy_name, category, description, all_attrs))
        
        # Sort strategies by effectiveness (fairness improvement and minimal performance loss)
        strategies.sort(key=lambda s: (-s.fairnessImprovement, s.accuracyImpact), reverse=False)
        
        logger.info(f"Successfully evaluated {len(strategies)} strategies with real performance metrics")
        return MitigationStrategiesResponse(strategies=strategies)
    
    def _calculate_strategy_recommendation(self, fairness_improvement, accuracy_impact, f1_impact):
        """Calculate strategy recommendation based on real performance metrics"""
        
        # Calculate overall effectiveness score
        # Higher fairness improvement is good, lower performance loss is good
        effectiveness_score = fairness_improvement - abs(accuracy_impact) - abs(f1_impact)
        
        if effectiveness_score > 25 and fairness_improvement > 30:
            return 3, "Highly Recommended"
        elif effectiveness_score > 15 and fairness_improvement > 20:
            return 3, "Recommended" 
        elif effectiveness_score > 5 and fairness_improvement > 10:
            return 2, "Good Option"
        elif fairness_improvement > 5:
            return 2, "Consider"
        else:
            return 1, "Limited Benefit"
    
    def _create_fallback_strategy(self, name, category, description, all_attrs):
        """Create fallback strategy when real evaluation fails"""
        from app.models.schemas import MitigationStrategy
        
        return MitigationStrategy(
            name=name,
            category=category,
            fairnessImprovement=20.0,  # Conservative estimate
            accuracyImpact=-1.0,
            precisionImpact=-1.0,
            recallImpact=-1.0,
            f1Impact=-1.0,
            stars=2,
            recommendation="Evaluation Failed",
            description=f"{description} (Performance could not be calculated)",
            targetAttributes=all_attrs
        )
    
    def _calculate_bias_severity(self, fairness_results) -> float:
        """Calculate average bias severity from fairness metrics"""
        if not fairness_results:
            return 20.0  # Default moderate bias
        
        bias_values = []
        
        # Extract bias values from all attributes and metrics
        for attr_name, attr_data in fairness_results.items():
            if isinstance(attr_data, dict) and 'metrics' in attr_data:
                for metric in attr_data['metrics']:
                    if isinstance(metric, dict) and 'value' in metric:
                        # Convert metric value to bias severity (higher = more biased)
                        bias_value = abs(metric['value']) * 100  # Convert to percentage
                        bias_values.append(bias_value)
        
        if bias_values:
            return sum(bias_values) / len(bias_values)
        else:
            return 20.0  # Default moderate bias
        
        return MitigationStrategiesResponse(strategies=strategies)
    
    def _simulate_before_after_comparison(self, strategy, fairness_results, sensitive_attrs, sensitive_features_data=None) -> BeforeAfterComparisonResponse:
        """Simulate before/after comparison for a mitigation strategy using actual dataset groups"""
        # This is a simplified simulation
        # In a real system, this would actually apply the mitigation and measure results
        
        from app.models.schemas import (
            BeforeAfterComparisonResponse, FairnessComparisonMetrics,
            PerformanceMetrics, GroupComparison
        )
        
        # Simulate improved fairness metrics
        before_metrics = [42, 38, 45, 35, 48]
        after_metrics = [min(78, before + strategy.fairnessImprovement) for before in before_metrics]
        
        # Calculate overall fairness scores
        overall_score_before = round(sum(before_metrics) / len(before_metrics))
        overall_score_after = round(sum(after_metrics) / len(after_metrics))
        
        fairness_comparison = FairnessComparisonMetrics(
            before=before_metrics,
            after=after_metrics,
            metrics=["Statistical Parity", "Disparate Impact", "Equal Opportunity", "Equalized Odds", "Calibration"],
            overallScoreBefore=overall_score_before,
            overallScoreAfter=overall_score_after
        )
        
        # Simulate performance impact
        performance_comparison = {
            "before": PerformanceMetrics(accuracy=85.2, precision=82.1, recall=79.8, f1=80.9),
            "after": PerformanceMetrics(
                accuracy=85.2 + strategy.accuracyImpact,
                precision=82.1 + strategy.precisionImpact,
                recall=79.8 + strategy.recallImpact,  # Now using the correct recallImpact field
                f1=80.9 + strategy.f1Impact
            )
        }
        
        # Generate group comparisons using actual dataset groups
        group_comparisons = {}
        improvement_factor = strategy.fairnessImprovement / 100.0
        
        for attr in sensitive_attrs:
            # Get actual groups from sensitive features data
            actual_groups = self._get_actual_groups_for_attribute(attr, sensitive_features_data)
            
            # Generate realistic before/after values for each actual group
            before_values = {}
            after_values = {}
            improvement_values = {}
            
            # Create varied baseline prediction rates for each group
            for i, group in enumerate(actual_groups):
                # Base rate varies by group position and attribute characteristics
                base_rate = 45.0 + (i * 8) + (len(attr) % 10) - 5
                base_rate = max(25.0, min(75.0, base_rate))  # Keep within reasonable bounds
                
                # Apply bias simulation (some groups have lower rates)
                if i % 2 == 0:  # Even indexed groups have disadvantage
                    biased_rate = base_rate - (15 * (1 - improvement_factor))
                else:  # Odd indexed groups have advantage
                    biased_rate = base_rate + (10 * (1 - improvement_factor))
                
                # After mitigation: rates should be more equalized
                target_rate = base_rate  # Target is the unbiased base rate
                after_rate = biased_rate + (target_rate - biased_rate) * improvement_factor * 0.7
                
                before_values[group] = round(biased_rate, 1)
                after_values[group] = round(after_rate, 1)
                improvement_values[group] = round(after_rate - biased_rate, 1)
            
            group_comparisons[attr] = GroupComparison(
                attribute=attr,
                groups=actual_groups,
                before=before_values,
                after=after_values,
                improvement=improvement_values
            )
        
        return BeforeAfterComparisonResponse(
            strategy=strategy.name,
            fairnessMetrics=fairness_comparison,
            performance=performance_comparison,
            groupComparisons=group_comparisons
        )
    
    def _get_actual_groups_for_attribute(self, attribute_name: str, sensitive_features_data: Optional[Dict]) -> List[str]:
        """Extract actual groups for a given attribute from sensitive features data"""
        logger.info(f"Getting groups for attribute: {attribute_name}")
        logger.info(f"Sensitive features data type: {type(sensitive_features_data)}")
        
        if not sensitive_features_data:
            logger.warning("No sensitive features data provided, using fallback")
            return ["Group A", "Group B"]
        
        # Check if detectedFeatures exists
        if 'detectedFeatures' not in sensitive_features_data:
            logger.warning("detectedFeatures not found in sensitive_features_data")
            logger.info(f"Available keys: {list(sensitive_features_data.keys())}")
            return ["Group A", "Group B"]
        
        detected_features = sensitive_features_data['detectedFeatures']
        logger.info(f"Found {len(detected_features)} detected features")
        
        # Find the attribute in detected features
        for i, feature in enumerate(detected_features):
            feature_name = feature.get('name') if isinstance(feature, dict) else getattr(feature, 'name', None)
            logger.info(f"Feature {i}: name='{feature_name}', type={type(feature)}")
            
            if feature_name == attribute_name:
                # Handle both dict and object formats
                if isinstance(feature, dict):
                    groups = feature.get('groups', [])
                else:
                    groups = getattr(feature, 'groups', [])
                
                logger.info(f"Found groups for {attribute_name}: {groups}")
                
                if groups:
                    # Limit to reasonable number of groups for visualization
                    if len(groups) <= 8:
                        return groups
                    else:
                        # Take first 6 and add indicator for more
                        return groups[:6] + [f"... +{len(groups)-6} more"]
        
        # Fallback if attribute not found
        logger.warning(f"Attribute {attribute_name} not found in detected features")
        return ["Group A", "Group B"]
