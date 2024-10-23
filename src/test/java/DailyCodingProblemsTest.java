import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.monke.DailyCodingProblems;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;


public class DailyCodingProblemsTest {

    DailyCodingProblems pb;

    /** Shortener method to create a non-immutable ArrayList of unknown type and length. */
    @SafeVarargs
    public static <T> List<T> setList(T... args) {
        return Arrays.stream(args).collect(Collectors.toList());
    }

    @BeforeEach
    public void init() {
        this.pb = new DailyCodingProblems();
    }

    @Test
    @DisplayName("twoSum")
    public void twoSumTest() {
        assertAll(
            () -> assertTrue(pb.twoSum(setList(10, 15, 3, 7), 17)),
            () -> assertTrue(pb.twoSum(setList(10, 15, 3, 7, 15, 46, 1, 25, 21, 11), 31)),
            () -> assertTrue(pb.twoSum(setList(10, 15, 3, 7, 7), 14)),
            () -> assertFalse(pb.twoSum(setList(10, 15, 3, 7), 21))
        );
    }

    @Test
    @DisplayName("exclusiveProducts")
    public void exclusiveProductsTest() {
        assertAll(
            () -> assertEquals(setList(2, 3, 6), pb.exclusiveProducts(setList(3, 2, 1))),
            () -> assertEquals(setList(120, 60, 40, 30, 24), pb.exclusiveProducts(setList(1, 2, 3, 4, 5))),
            () -> assertEquals(setList(6, 0, 0), pb.exclusiveProducts(setList(0, 2, 3)))
        );
    }

    @Test
    @DisplayName("firstMissingPositive")
    public void firstMissingPositiveTest() {
        assertAll(
            () -> assertEquals(2, pb.firstMissingPositive(setList(3, 4, -1, 1))),
            () -> assertEquals(3, pb.firstMissingPositive(setList(1, 2, 0))),
            () -> assertEquals(5, pb.firstMissingPositive(setList(3, 4, 1, 2)))
        );
    }

    @Test
    @DisplayName("cons/car/cdr")
    public void consCarCdrTest() {
        assertAll(
            () -> assertEquals(2, pb.car(pb.cons(2, "3"))),
            () -> assertEquals("3", pb.cdr(pb.cons(2, "3"))),
            () -> assertEquals(BigDecimal.ZERO, pb.cdr(pb.cons("aaa", BigDecimal.ZERO)))
        );
    }

    @Test
    @DisplayName("numEncodings")
    public void numEncodingsTest() {
        assertAll(
            () -> assertEquals(0, pb.numEncodings("000")),
            () -> assertEquals(1, pb.numEncodings("101")),
            () -> assertEquals(1, pb.numEncodings("354")),
            () -> assertEquals(1, pb.numEncodings("44")),
            () -> assertEquals(1, pb.numEncodings("")),
            () -> assertEquals(3, pb.numEncodings("111"))
        );
    }

    @Test
    @DisplayName("staircase")
    public void staircaseTest() {
        assertAll(
            () -> assertEquals(5, pb.staircase(4, setList(1, 2))),
            () -> assertEquals(8, pb.staircase(5, setList(1, 2))),
            () -> assertEquals(5, pb.staircase(5, setList(1, 3, 5)))
        );
    }

    @Test
    @DisplayName("reservoirSampling [Manual]")
    public void reservoirSamplingTest() {
        List<Integer> stream = IntStream.range(0, 100).boxed().toList();

        IntStream.range(0, 10).forEach((i) ->
            System.out.println(pb.reservoirSampling(stream)));
    }

    @Test
    @DisplayName("log")
    public void logTest() {
        DailyCodingProblems.Log<String> log = new DailyCodingProblems.Log<>(10);
        String logs = "abcdefghijkl";

        logs.chars()
            .mapToObj(ch -> (char) ch)
            .map(String::valueOf)
            .forEach(log::record);

        assertAll(
            () -> assertEquals("j", log.getLast(3)),
            () -> assertEquals("l", log.getLast(1)),
            () -> assertEquals("d", log.getLast(9))
        );
    }

    @Test
    @DisplayName("maxOfSubArrays")
    public void maxOfSubArraysTest() {
        assertAll(
            () -> assertEquals(setList(10, 7, 8, 8), pb.maxOfSubArrays(setList(10, 5, 2, 7, 8, 7), 3))
            // () -> assertEquals(setList(10, 10, 8, 8), pb.maxOfSubArrays(setList(2, 10, 5, 7, 8, 7), 3)) //TODO
        );
    }

    //TODO
    @Test
    @DisplayName("intersection")
    public void intersectionTest() {
        LinkedList<Integer> lList1 = new LinkedList<>(setList(3, 7, 8, 10, 15, 6));
        LinkedList<Integer> lList2 = new LinkedList<>(setList(99, 1, 8, 10));

//        assertAll(
//            () -> assertEquals(8, pb.intersection(lList1, lList2))
//        );
    }

    @Test
    @DisplayName("maxOverlapping")
    public void maxOverlappingTest() {
        assertAll(
            () -> assertEquals(2, pb.maxOverlapping(setList(new int[]{30, 75}, new int[]{0, 50}, new int[]{60, 150}))),
            () -> assertEquals(1, pb.maxOverlapping(setList(new int[]{0, 10}, new int[]{11, 20}, new int[]{21, 30}))),
            () -> assertEquals(3, pb.maxOverlapping(setList(
                new int[]{0, 10},
                new int[]{9, 20},
                new int[]{10, 30},
                new int[]{31, 40},
                new int[]{41, 50},
                new int[]{0, 60})))
        );
    }
}
