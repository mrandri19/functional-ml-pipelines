{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
-- TODO(Andrea): why does the feature engineering transformer even care about whether
-- the type of preds?
-- How do I make it generic over it? ... using Rank2Types
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

-- | A database-like list of columns.
data DataFrame = DataFrame
  { x :: [Float],
    y :: [String]
  }
  deriving (Show)

-------------------------------------------------------------------------------------

data DatasetKind = Raw | Clean

-- Phantom types require a `newtype`. Using `type` just is not enough
newtype Dataset (d :: DatasetKind) = Dataset
  { datasetDataFrame :: DataFrame
  }
  deriving (Show)

-- | Load a dataset from disk, as is.
load :: Dataset Raw
load = Dataset $ DataFrame [1.0, 2.0] ["3.0", "4.0"]

-- | Clean up a raw dataset, e.g. by reducing its memory usage and dropping duplicates
clean :: Dataset Raw -> Dataset Clean
clean !rawDataset = Dataset (datasetDataFrame rawDataset)

-------------------------------------------------------------------------------------

data FeaturesKind = WRawFeatures | WFeatures

data TargetKind = NoTarget | WRawTarget | WTarget

data PredsKind = NoPreds | WRawPreds | WPreds

newtype Fold (f :: FeaturesKind) (t :: TargetKind) (p :: PredsKind) = Fold
  { foldDataFrame :: DataFrame
  }
  deriving (Show)

holdout ::
  Dataset Clean ->
  ( Fold WRawFeatures WRawTarget NoPreds,
    Fold WRawFeatures WRawTarget NoPreds
  )
holdout !cleanDataset =
  let df = datasetDataFrame cleanDataset
      xs = x df
      ys = y df
      fold1 = Fold $ DataFrame [head xs] [head ys]
      fold2 = Fold $ DataFrame (tail xs) (tail ys)
   in (fold1, fold2)

-------------------------------------------------------------------------------------

type FETransformer =
  forall trans_target trans_preds.
  Fold WRawFeatures trans_target trans_preds ->
  Fold WFeatures trans_target trans_preds

type FETrainer =
  forall train_preds.
  Fold WRawFeatures WRawTarget train_preds ->
  FETransformer

trainFE :: FETrainer
trainFE !_fold =
  let feTransformer fold = Fold $ foldDataFrame fold
   in feTransformer

-------------------------------------------------------------------------------------

type TargetFETransformer f p = Fold f WRawTarget p -> Fold f WTarget p

type TargetInverseFETransformer f p = Fold f WTarget p -> Fold f WRawTarget p

type PredsInverseFETransformer f t = Fold f t WPreds -> Fold f t WRawPreds

type TargetFETrainer =
  forall
    train_feats
    train_preds
    --
    tfe_trans_feats
    tfe_trans_preds
    --
    tife_trans_feats
    tife_trans_preds
    --
    pife_trans_feats
    pife_trans_target.
  Fold train_feats WRawTarget train_preds ->
  ( TargetFETransformer tfe_trans_feats tfe_trans_preds,
    TargetInverseFETransformer tife_trans_feats tife_trans_preds,
    PredsInverseFETransformer pife_trans_feats pife_trans_target
  )

trainTargetFE :: TargetFETrainer
trainTargetFE !_fold = (targetFETransformer, targetInverseFETransformer, predsInverseFETransformer)
  where
    targetFETransformer fold = Fold $ foldDataFrame fold
    targetInverseFETransformer fold = Fold $ foldDataFrame fold
    predsInverseFETransformer fold = Fold $ foldDataFrame fold

-------------------------------------------------------------------------------------

type Model = forall t p. Fold WFeatures t p -> Fold WFeatures t WPreds

type ModelTrainer = forall p. Fold WFeatures WTarget p -> Model

trainModel :: ModelTrainer
trainModel !_fold = model
  where
    model fold =
      let df = foldDataFrame fold
          xs = x df
          ys = y df
          preds = map (show . (\y -> read @Float y + 0.5)) ys
          newDf = DataFrame xs preds
       in Fold newDf

-------------------------------------------------------------------------------------

predictFolds ::
  forall train_preds test_preds.
  Fold WRawFeatures WRawTarget train_preds ->
  Fold WRawFeatures WRawTarget test_preds ->
  FETrainer ->
  TargetFETrainer ->
  ModelTrainer ->
  Fold WFeatures WRawTarget WRawPreds
predictFolds
  !trainFold
  !testFold
  !feTrainer
  !targetFETrainer
  !modelTrainer =
    let feTransformer = feTrainer trainFold
        trainFold' = feTransformer trainFold
        testFold' = feTransformer testFold

        ( targetFETransformer,
          targetInverseFETransformer,
          predsInverseFETransformer
          ) = targetFETrainer trainFold
        trainFold'' = targetFETransformer trainFold'
        testFold'' = targetFETransformer testFold'

        model = modelTrainer trainFold''
        preds'' = model testFold''
        preds' = predsInverseFETransformer preds''
        preds = targetInverseFETransformer preds'
     in preds

evaluate :: Fold f WRawTarget WRawPreds -> Float
evaluate !predictFold = 0.5

-------------------------------------------------------------------------------------

main :: IO ()
main = do
  let !rawDataset = load
  let !cleanDataset = clean rawDataset

  let (trainFold, testFold) = holdout cleanDataset

  let !testFoldWPreds =
        predictFolds
          trainFold
          testFold
          trainFE
          trainTargetFE
          trainModel

  print rawDataset
  print testFold
  print testFoldWPreds

  let !score = evaluate testFoldWPreds

  print score
