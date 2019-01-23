//! Manipulation of vectors and inequalities in a flag algebra.

use crate::flag::Flag;
use crate::operator::*;
use crate::prettyprint::Expr;
use ndarray::{Array, Array1, ArrayBase, ScalarOperand};
use num::{Integer, Num, NumCast, ToPrimitive};
use sprs::{CsMat, CsMatView};
use std::fmt::*;
use std::ops::*;

/// An element of a flag algebra.
#[derive(Clone, Debug)]
pub struct QFlag<N, F> {
    /// Basis of the space where the element lives. This corresponds to the size and type of the flags.
    pub basis: Basis<F>,
    /// The vector of the element in the corresponding basis is `(1/self.scale).self.data`.
    pub data: Array1<N>,
    /// Scaling factor of the vector.
    pub scale: u64,
    /// Expression recording how the vector was computed.
    pub expr: Expr,
}

// equality for QFlags
impl<N, F> PartialEq for QFlag<N, F>
where
    N: Num + NumCast + Copy,
    F: Flag,
{
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.basis, other.basis);
        assert_eq!(self.data.len(), other.data.len());
        //
        let s1 = N::from(self.scale).unwrap();
        let s2 = N::from(other.scale).unwrap();
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&x, &y)| x * s2 == y * s1)
    }
}

// ==================== operarions on Qflags ===========

// The arithmetic to put two flags on same denominator
fn matching_scales<N>(scale1: u64, scale2: u64) -> (N, N, u64)
where
    N: NumCast,
{
    let scale = scale1.gcd(&scale2);
    let c1 = N::from(scale1 / scale).unwrap();
    let c2 = N::from(scale2 / scale).unwrap();
    (c1, c2, scale)
}

impl<'a, N, F> Add for &'a QFlag<N, F>
where
    N: Clone + NumCast + Num + ScalarOperand,
    F: Flag,
{
    type Output = QFlag<N, F>;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.basis, other.basis);
        assert_eq!(self.data.len(), other.data.len());
        let (a1, a2, scale) = matching_scales::<N>(self.scale, other.scale);
        QFlag {
            basis: self.basis,
            data: &self.data * a1 + &other.data * a2,
            scale,
            expr: Expr::add(self.expr.clone(), other.expr.clone()),
        }
    }
}

impl<N, F> Add for QFlag<N, F>
where
    N: Clone + Num + NumCast + ScalarOperand,
    F: Flag,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<'a, N, F> Sub for &'a QFlag<N, F>
where
    N: Clone + Num + NumCast + ScalarOperand,
    F: Flag,
{
    type Output = QFlag<N, F>;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.basis, other.basis);
        assert_eq!(self.data.len(), other.data.len());
        let (a1, a2, scale) = matching_scales::<N>(self.scale, other.scale);
        QFlag {
            basis: self.basis,
            data: &self.data * a1 - &other.data * a2,
            scale,
            expr: Expr::sub(self.expr.clone(), other.expr.clone()),
        }
    }
}

impl<N, F> Sub for QFlag<N, F>
where
    N: Clone + Num + NumCast + ScalarOperand,
    F: Flag,
{
    type Output = QFlag<N, F>;

    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<N, F> Neg for QFlag<N, F>
where
    N: Clone + Neg<Output = N>,
{
    type Output = QFlag<N, F>;

    fn neg(self) -> Self::Output {
        QFlag {
            basis: self.basis,
            data: -self.data,
            scale: self.scale,
            expr: self.expr.neg(),
        }
    }
}

impl<'a, N, F> Neg for &'a QFlag<N, F>
where
    N: Clone + Neg<Output = N>,
{
    type Output = QFlag<N, F>;

    fn neg(self) -> Self::Output {
        QFlag {
            basis: self.basis,
            data: -self.data.clone(),
            scale: self.scale,
            expr: self.expr.clone().neg(),
        }
    }
}

// Right scalar multiplication (it is not possible to implement it on left)
impl<S, N, F> Mul<S> for QFlag<N, F>
where
    N: Num + ScalarOperand + NumCast + Display,
    F: Flag,
    S: ToPrimitive,
{
    type Output = QFlag<N, F>;

    fn mul(self, rhs: S) -> Self::Output {
        let rhs_n = N::from(rhs).unwrap();
        QFlag {
            expr: Expr::mul(self.expr.clone(), Expr::num(&rhs_n)),
            basis: self.basis,
            data: self.data * rhs_n,
            scale: self.scale,
        }
    }
}

// =================
fn quadratic_form<N, I>(lhs: &Array1<N>, matrix: &CsMat<I>, rhs: &Array1<N>) -> N
where
    N: Copy + Num + NumCast,
    I: Num + Copy + ToPrimitive,
{
    assert_eq!(lhs.len(), matrix.rows());
    assert_eq!(rhs.len(), matrix.cols());
    let mut res = N::zero();
    for (&v, (i, j)) in matrix.iter() {
        res = res + (N::from(v).unwrap() * lhs[i] * rhs[j]);
    }
    res
}

fn vector_matrix_mul<N, I>(matrix: &CsMatView<I>, vec: &Array1<N>) -> Array1<N>
where
    N: Copy + Num + AddAssign + NumCast,
    I: Num + Copy + ToPrimitive,
{
    assert_eq!(vec.len(), matrix.cols());
    let mut res = Array1::zeros(matrix.rows());
    for (&v, (i, j)) in matrix.iter() {
        res[i] += N::from(v).unwrap() * vec[j];
    }
    res
}

fn multiply<N, I>(lhs: &Array1<N>, table: &[CsMat<I>], rhs: &Array1<N>) -> Array1<N>
where
    N: Copy + Num + NumCast,
    I: Num + Copy + ToPrimitive,
{
    let mut res = Array1::<N>::zeros(table.len());
    for (i, matrix) in table.iter().enumerate() {
        res[i] = quadratic_form(lhs, matrix, rhs);
    }
    res
}

impl<N, F> QFlag<N, F>
where
    N: Copy + Num + Default + AddAssign + NumCast,
    F: Flag,
{
    fn raw_expand(&self, operator: &CsMat<u64>, outbasis: Basis<F>, denom: u64) -> Self {
        QFlag {
            basis: outbasis,
            data: vector_matrix_mul(&operator.view(), &self.data),
            scale: self.scale * denom,
            expr: self.expr.clone(),
        }
    }

    fn raw_multiply(&self, table: &[CsMat<u64>], other: &Self, denom: u64) -> Self {
        assert_eq!(self.basis.t, other.basis.t);
        QFlag {
            basis: self.basis * other.basis,
            data: multiply(&self.data, table, &other.data),
            scale: self.scale * denom * other.scale,
            expr: Expr::mul(self.expr.clone(), other.expr.clone()),
        }
    }
}

fn raw_untype<N>(
    input: &Array1<N>,
    untype_flag: &[usize],
    untype_count: &[u64],
    outbasis_size: usize,
) -> Array1<N>
where
    N: Copy + Num + Default + NumCast + AddAssign,
{
    assert_eq!(input.len(), untype_flag.len());
    assert_eq!(input.len(), untype_count.len());
    let mut output = Array1::<N>::zeros(outbasis_size);
    for (i, &v) in input.iter().enumerate() {
        output[untype_flag[i]] += v * N::from(untype_count[i]).unwrap()
    }
    output
}

impl<N, F> QFlag<N, F>
where
    N: Copy + Num + Default + NumCast + AddAssign,
    F: Flag,
{
    fn raw_untype(
        &self,
        untype_flag: &[usize],
        untype_count: &[u64],
        outbasis: Basis<F>,
        outbasis_size: usize,
        denom: u64,
    ) -> Self {
        QFlag {
            basis: outbasis,
            data: raw_untype(&self.data, untype_flag, untype_count, outbasis_size),
            scale: self.scale * denom,
            expr: self.expr.clone().unlab(),
        }
    }
}

impl<N, F> QFlag<N, F>
where
    N: Copy + Num + Default + NumCast + AddAssign,
    F: Flag,
{
    /// Projection to a basis of larger flag.
    pub fn expand(&self, outbasis: Basis<F>) -> Self {
        let subflag = SubflagCount::from_to(self.basis, outbasis);
        self.raw_expand(&subflag.get(), outbasis, subflag.denom())
    }
    /// Unlabeling operator 〚.〛 to the flag algebra of completly unlabeled flags.
    pub fn untype(&self) -> Self {
        let unlabeling = Unlabeling::<F>::total(self.basis.t);
        let size = self.basis.size;
        let unlab_flag = UnlabelingFlag { unlabeling, size };
        let unlab_count = UnlabelingCount { unlabeling, size };
        let outbasis = self.basis.with_type(Type::empty());
        self.raw_untype(
            &unlab_flag.get(),
            &unlab_count.get(),
            outbasis,
            outbasis.get().len(),
            unlab_count.denom(),
        )
    }
}

impl<'a, N, F> Mul for &'a QFlag<N, F>
where
    N: Num + Copy + AddAssign + Default + NumCast,
    F: Flag,
{
    type Output = QFlag<N, F>;

    fn mul(self, other: Self) -> QFlag<N, F> {
        let split = SplitCount::from_input(&self.basis, &other.basis);
        self.raw_multiply(&split.get(), other, split.denom())
    }
}

impl<N, F> Mul for QFlag<N, F>
where
    N: Num + Copy + AddAssign + Default + NumCast,
    F: Flag,
{
    type Output = Self;

    fn mul(self, other: Self) -> QFlag<N, F> {
        &self * &other
    }
}

// ===============
impl<N, F> QFlag<N, F>
where
    N: Num + NumCast + Display,
    F: Flag,
{
    /// Return the inequality "`self` ≥ `n`".
    pub fn at_least(self, x: N) -> Ineq<N, F> {
        Ineq {
            meta: IneqMeta {
                basis: self.basis,
                flag_expr: self.expr,
                bound_expr: Expr::num(&x),
            },
            data: vec![IneqData {
                flag: self.data,
                bound: x * N::from(self.scale).unwrap(),
            }],
        }
    }
    /// Return the inequality "`self` ≤ `n`".
    pub fn at_most(self, n: N) -> Ineq<N, F>
    where
        N: Clone + Neg<Output = N>,
    {
        (-self).at_least(-n)
    }
    /// Return the inequality "`self` ≥ `0`".
    pub fn non_negative(self) -> Ineq<N, F>
    where
        N: Num,
    {
        self.at_least(N::zero())
    }
}

/// Return the inequalities expressing that the sum of the flags of `basis`
/// is equal to one.
pub fn total_sum_is_one<N, F>(basis: Basis<F>) -> Ineq<N, F>
where
    F: Flag,
    N: Num + Clone + Neg<Output = N> + NumCast + Display,
{
    let one: QFlag<N, F> = basis.one();
    one.at_least(N::one()).equality()
}

/// Return the inequalities expressing that the flags of `basis`
/// are larger than zero.
pub fn flags_are_nonnegative<N, F>(basis: Basis<F>) -> Ineq<N, F>
where
    F: Flag,
    N: Num + Clone + Neg<Output = N>,
{
    let n = basis.get().len();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let mut flag = Array::zeros(n);
        flag[i] = N::one();
        data.push(IneqData {
            flag,
            bound: N::zero(),
        })
    }
    let meta = IneqMeta {
        basis,
        flag_expr: Expr::Num(String::from("x")),
        bound_expr: Expr::Zero,
    };
    Ineq { meta, data }
}

//============
#[derive(Clone, Debug)]
/// Contains informations about a set of inequalities of a flag algebra.
pub struct IneqMeta<F> {
    /// Basis in which the inequality is expressed.
    /// This correspond to the type and size of the flags.
    pub basis: Basis<F>,
    /// Expression recording how the left sides of the inequalities where constructed.
    pub flag_expr: Expr,
    /// Expression recording how the right sides where constructed.
    pub bound_expr: Expr,
}

impl<F: Flag> IneqMeta<F> {
    fn opposite(self) -> Self {
        IneqMeta {
            basis: self.basis,
            flag_expr: self.flag_expr.neg(),
            bound_expr: self.bound_expr.neg(),
        }
    }
    fn one_sided_expr(self) -> Expr {
        Expr::sub(self.flag_expr, self.bound_expr)
    }

    fn multiply(self, rhs_basis: &Basis<F>, rhs_expr: Expr) -> Self {
        IneqMeta {
            basis: self.basis * *rhs_basis,
            flag_expr: Expr::mul(self.one_sided_expr(), rhs_expr),
            bound_expr: Expr::Zero,
        }
    }

    fn untype(self) -> Self {
        IneqMeta {
            basis: self.basis.with_type(Type::empty()),
            flag_expr: Expr::unlab(self.flag_expr),
            bound_expr: self.bound_expr,
        }
    }
}

#[derive(Clone, Debug)]
/// Contains the vector and the bound of one inequality in a flag algebra.
/// This inequality has the form `self.flag  ≥ self.bound`.
/// Expression recording how the left sides where constructed.
pub struct IneqData<N> {
    /// Vector of the left side in the corresponding flag basis.
    pub flag: Array1<N>,
    /// Number on the right side of the inequality.
    pub bound: N,
}

impl<N> IneqData<N>
where
    N: Num + Clone,
{
    fn opposite(self) -> Self
    where
        N: Neg<Output = N>,
    {
        IneqData {
            flag: -self.flag,
            bound: -self.bound,
        }
    }
    fn one_sided(self) -> Self
    where
        N: Copy + ScalarOperand + SubAssign,
    {
        let mut flag = self.flag;
        if self.bound != N::zero() {
            flag.sub_assign(self.bound);
        }
        IneqData {
            flag,
            bound: N::zero(),
        }
    }
    fn untype(
        &self,
        untype_flag: &[usize],
        untype_count: &[u64],
        denom: u64,
        outbasis_size: usize,
    ) -> Self
    where
        N: Copy + ScalarOperand + NumCast + AddAssign + Default,
    {
        IneqData {
            flag: raw_untype(&self.flag, untype_flag, untype_count, outbasis_size),
            bound: self.bound * N::from(denom).unwrap(),
        }
    }

    fn multiply(&self, table: &[CsMat<u64>], g: &Array1<N>) -> Self
    where
        N: Copy + Default + SubAssign + AddAssign + ScalarOperand + NumCast,
    {
        let flag = multiply(&self.clone().one_sided().flag, table, g); // tbo
        IneqData {
            flag,
            bound: N::zero(),
        }
    }

    fn multiply_by_all(self, table: &[CsMat<u64>], acc: &mut Vec<Self>)
    where
        N: Copy + AddAssign + SubAssign + ScalarOperand + NumCast,
    {
        //
        let one_sided = self.one_sided();
        let pre_result: Vec<_> = table
            .iter()
            .map(|m| vector_matrix_mul(&m.transpose_view(), &one_sided.flag))
            .collect();
        if let Some(other_size) = pre_result.first().map(|v| v.len()) {
            for i in 0..other_size {
                let vec: Vec<_> = pre_result.iter().map(|x| x[i]).collect();
                let ineq_data = IneqData {
                    flag: ArrayBase::from_vec(vec),
                    bound: N::zero(),
                };
                acc.push(ineq_data)
            }
        }
    }
}

#[derive(Clone, Debug)]
/// A set of bounds on elements of a flag algebra.
///
/// This correpond to a set of inequalities constructed in a similar way.
pub struct Ineq<N, F> {
    /// Common information about the set of inequalities.
    pub meta: IneqMeta<F>,
    /// List of data of the inequalities in the set.
    pub data: Vec<IneqData<N>>,
}

impl<N, F> Ineq<N, F>
where
    N: Num + Neg<Output = N> + Clone,
    F: Flag + Clone,
{
    /// If self is "`f ≥ x`", returns "`f ≤ x`".
    pub fn opposite(self) -> Self {
        Ineq {
            meta: self.meta.opposite(),
            data: self.data.into_iter().map(|x| x.opposite()).collect(),
        }
    }

    // FIXME: incorrect metadata
    /// If self is "`f ≥ x`", returns "`f = x`".
    pub fn equality(mut self) -> Self {
        let mut opposite_data = self.clone().opposite().data;
        self.data.append(&mut opposite_data);
        self
    }
}

impl<N, F> Ineq<N, F>
where
    N: Num + Copy + Default + AddAssign + NumCast + ScalarOperand,
    F: Flag + Clone,
{
    /// If self is "`f` ≥ `x`", return the projection "`〚f〛 ≥ x`".
    pub fn untype(self) -> Self {
        let unlabeling = Unlabeling::<F>::total(self.meta.basis.t);
        let size = self.meta.basis.size;
        let unlab_f = (UnlabelingFlag { unlabeling, size }).get();
        let unlab_c = (UnlabelingCount { unlabeling, size }).get();
        let denom = (UnlabelingCount { unlabeling, size }).denom();
        let basis = self.meta.basis.with_type(Type::empty());
        let basis_size = basis.get().len();
        //
        let mut data = Vec::new();
        for i in self.data.iter() {
            let f = i.untype(&unlab_f, &unlab_c, denom, basis_size);
            data.push(f)
        }
        Ineq {
            meta: self.meta.untype(),
            data,
        }
    }
}

impl<N, F> Ineq<N, F>
where
    N: Num + Copy + Default + AddAssign + SubAssign + ScalarOperand + NumCast,
    F: Flag + Clone,
{
    /// If self is "`f` ≥ `x`", return the inequality "`f*g ≥ x.g`".
    pub fn multiply_by_qflag(&self, g: &QFlag<N, F>) -> Self {
        let split = SplitCount::from_input(&self.meta.basis, &g.basis);
        let table = split.get();
        //
        let mut data = Vec::new();
        for i in self.data.iter() {
            data.push(i.multiply(&table, &g.data));
        }
        Ineq {
            data,
            meta: self.meta.clone().multiply(&g.basis, g.expr.clone()),
        }
    }
    /// If self is "`f` ≥ `x`", return the set of inequalities "`f*g ≥ x.g`",
    /// where `g` is chosen such that `f*g` is a vector of `outbasis`.
    pub fn multiply_by_all(self, outbasis: Basis<F>) -> Self {
        let b = outbasis / self.meta.basis;
        let splitcount = SplitCount::from_input(&self.meta.basis, &b);
        let table = splitcount.get();
        //
        let mut data = Vec::new();
        for ineq in self.data {
            ineq.multiply_by_all(&table, &mut data)
        }
        //
        Ineq {
            data,
            meta: self.meta.multiply(&b, Expr::Var(0)),
        }
    }
    /// If self is "`f` ≥ `x`", return the set of inequalities "`〚f*g〛 ≥ x.〚g〛`",
    /// where `g` is chosen such that `〚f*g〛` is a vector of `outbasis`.
    pub fn multiply_and_unlabel(self, outbasis: Basis<F>) -> Self {
        assert!(outbasis.t == Type::empty());
        let unlabeling = Unlabeling::total(self.meta.basis.t);
        let other = outbasis.with_type(self.meta.basis.t) / self.meta.basis;
        let splitcount = SplitCount::from_input(&self.meta.basis, &other);
        let operator = MulAndUnlabeling::new(splitcount, unlabeling);
        //
        let table = operator.get();
        //
        let mut data = Vec::new();
        //
        for ineq in self.data {
            ineq.multiply_by_all(&table, &mut data)
        }
        Ineq {
            data,
            meta: self.meta.multiply(&other, Expr::Var(0)).untype(),
        }
    }
}

/// Return the vector corresponding to the unlabeled flag f
pub fn flag<N, F>(f: &F) -> QFlag<N, F>
where
    N: Num + Clone,
    F: Flag,
{
    Basis::new(f.size()).flag(f)
}